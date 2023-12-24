#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import boundary_loss_func, l1_loss, sigmoid_anneal, ssim, weighted_loss_sum
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from  torchvision.transforms import GaussianBlur
from ray import tune
from ray.train import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, hparams):
    '''
        Main training script for Gaussian Splatting
        
        :param dataset: Dataset object for train and test
        :param opt: Optimization parameters
        :param pipe: Pipeline for rendering the Gaussian model
        :param testing_iterations: Iterations to test on 
        :param saving_iterations: Iterations to save the model on
        :param checkpoint_iterations: Iterations to create model checkpoints on
        :param checkpoint: If not None, the path to load a checkpoint model from
        :param debug_from: Iteration to debug from
        :param hparams: Training loss hyperparameters
        
        :returns: Model's final training results
    '''
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    max_iterations = opt.iterations + 1
    results = None
    for iteration in range(first_iter, max_iterations):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, mask, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["mask"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        ######Loss#######
        #################
        gt_image = viewpoint_cam.original_image.cuda()
        
        gt_mask = viewpoint_cam.original_depth_mask.cuda()
        gt_mask_bin = gt_mask.clone()
        gt_mask_bin[gt_mask > 0.0] = 1.0
              
        boundary_loss_raw = boundary_loss_func(mask, gt_mask_bin, hparams['boundary_dice_prop'], hparams['boundary_bce_prop'])
        
        Ll1_image = l1_loss(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * Ll1_image + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        annealed_image_loss = sigmoid_anneal(loss_image, iteration, max_iterations, hparams['img_sigmoid_p'], hparams['img_sigmoid_k'])

        Ll1_depth = l1_loss(depth, gt_mask)        
        ssim_depth = 1.0 - ssim(depth, gt_mask)
        
        loss_depth = weighted_loss_sum([hparams['Ll1_depth'], hparams['ssim_depth'], hparams['boundary_depth']], [Ll1_depth, ssim_depth, boundary_loss_raw])
        annealed_depth_loss = sigmoid_anneal(loss_depth, iteration, max_iterations, hparams['depth_sigmoid_p'], hparams['depth_sigmoid_k'])
        
        scaling_loss = gaussians.scaling_reg_loss()
        annealed_scaling_loss = sigmoid_anneal(scaling_loss, iteration, max_iterations, hparams['depth_sigmoid_p'], hparams['depth_sigmoid_k'])

        loss = weighted_loss_sum([hparams['image_loss'], hparams['depth_loss'], hparams['scaling_loss']], [annealed_image_loss, annealed_depth_loss, annealed_scaling_loss])
        loss *= 10
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            results = training_report(tb_writer, iteration, Ll1_image, loss_depth, boundary_loss_raw, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), len(gaussians.get_xyz))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter: 
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.001, 0.01, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    return results

def prepare_output_and_logger(args):
    '''
        Create path for saving the model and a logger for tracking results
        
        :param args: arguments containing the model path
        
        :returns: tensorboard logger
    '''
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1_image, loss_mask, boundary_loss_raw, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, numPoints):
    '''
        Logs training progress with tensorboard
        :param tb_writer: Tensorboard logging object
        :param iteration: Current iteration in training
        :param Ll1_image: L1 loss for render
        :param loss_mask: Loss on mask of render
        :param boundary_loss_raw: Boundary loss for render
        :param loss: Overall oss
        :param l1_loss: L1 loss function
        :param elapsed: Elapsed time
        :param testing_iterations: Iterations to perform testing on
        :param scene: Scene object for the model and ground truth views
        :param renderFunc: Function for rendering the views of the model
        :param renderArgs: Arguments to pass to rendering function
        :param numPoints: Number of Gaussians in model
        
        :returns: A dictionary of training results at this point in training
    '''
    result = {}
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss_image', Ll1_image.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_mask', loss_mask.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/boundary_loss_raw', boundary_loss_raw.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                boundary_loss = 0.0
                l1_test_image = 0.0
                l1_test_mask = 0.0
                psnr_test_image = 0.0
                psnr_test_mask = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    depths = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["depth"], 0.0, 1.0)
                    mask = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["mask"], 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_depth_mask.to("cuda"), 0.0, 1.0)   
                    gt_mask_bin = (gt_mask > 0).float()
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_mask".format(viewpoint.image_name), mask[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_depths".format(viewpoint.image_name), depths[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask".format(viewpoint.image_name), gt_mask[None], global_step=iteration)
                    boundary_loss += boundary_loss_func(mask, gt_mask_bin, 1, 1).double()   
                    l1_test_image += l1_loss(image, gt_image).mean().double()
                    l1_test_mask += l1_loss(depths, gt_mask).mean().double()
                    psnr_test_image += psnr(image, gt_image).mean().double()
                    psnr_test_mask += psnr(depths, gt_mask).mean().double()
                psnr_test_image /= len(config['cameras'])
                psnr_test_mask /= len(config['cameras'])
                l1_test_image /= len(config['cameras'])          
                l1_test_mask /= len(config['cameras'])     
                boundary_loss /= len(config['cameras'])
                if config['name'] == 'test':
                    result["weighted_metric_sum"] = psnr_test_image.item() / 28 + psnr_test_mask.item() / 17 + (1 - boundary_loss.item())/0.7
                    result["psnr_test_image"] = psnr_test_image.item()
                    result["psnr_test_mask"] = psnr_test_mask.item()
                    result["boundary_loss"] = 1 - boundary_loss.item()
                    
                print("\n[ITER {}] Evaluating Image {}: L1 {} PSNR {} | {} points".format(iteration, config['name'], round(l1_test_image.item(), 4), round(psnr_test_image.item(), 4), numPoints))
                print("\n[ITER {}] Evaluating Mask {}: L1 {} PSNR {} BOUND {} | {} points".format(iteration, config['name'], round(l1_test_mask.item(), 4), round(psnr_test_mask.item(), 4), round(boundary_loss.item(), 4), numPoints))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_image', l1_test_image, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_mask', l1_test_mask, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_image', psnr_test_image, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_mask', psnr_test_mask, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - boundary_loss', boundary_loss, iteration)
                    

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/geometry_opacity_histogram", scene.gaussians.get_geometry_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
    return result

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--hparam_tune', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    
    # Hyperparameter optimization code
    if args.hparam_tune:
        def trainable(config):
            results = training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, config)
            return results
            
        algo = BayesOptSearch()
        algo = ConcurrencyLimiter(algo, max_concurrent=4)
        num_samples = 100
        
        # Define hyperparam search space
        search_space = {
            "boundary_dice_prop": tune.uniform(0.0, 1.0),
            "boundary_bce_prop": tune.uniform(0.0, 1.0),
            "img_sigmoid_p": tune.uniform(0.0, 1.1),
            "img_sigmoid_k": tune.uniform(-1.0, 1.0),
            'Ll1_depth': tune.uniform(0.0, 1.0),
            'ssim_depth': tune.uniform(0.0, 1.0),
            'boundary_depth': tune.uniform(0.0, 1.0),
            "depth_sigmoid_p": tune.uniform(0.0, 1.1),
            "depth_sigmoid_k": tune.uniform(-1.0, 1.0),
            'image_loss': tune.uniform(0.0, 1.0),
            'depth_loss': tune.uniform(0.0, 1.0),
            'scaling_loss': tune.uniform(0.0, 1.0),
        }
        
        tune_config = tune.TuneConfig(
            metric="weighted_metric_sum",
            mode="max",
            search_alg=algo,
            num_samples=num_samples,
        )

        # Start a Tune run and print the best result.
        trainable_with_resources = tune.with_resources(trainable, resources={"gpu": 1, "cpu": 8})
        
        tuner = tune.Tuner(
            trainable_with_resources,
            tune_config=tune_config,
            param_space=search_space,
            run_config=RunConfig(storage_path="~/Repos/gaussian-splatting/raytune_results_Nov13", name="hparam_tuning")
        )
        results = tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)
    
    # Train without hparam tuning
    else:
        # Opt after render bug fix
        hparams = {'boundary_dice_prop': 0.4318835235454354, 'boundary_bce_prop': 0.9052478919311929, 'img_sigmoid_p': 1.1, 'img_sigmoid_k': -0.5959912983264622, 'Ll1_depth': 0.6171146148837469, 'ssim_depth': 0.0, 'boundary_depth': 1.0, 'depth_sigmoid_p': 1.0259194345391112, 'depth_sigmoid_k': 0.835191693021905, 'image_loss': 0.14695685368823122, 'depth_loss': 1.0, 'scaling_loss': 1.0}
        training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, hparam)

    # All done
    print("\nTraining complete.")
