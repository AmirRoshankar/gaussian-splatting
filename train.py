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

from ast import List
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torchvision.transforms.functional import rgb_to_grayscale
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
import numpy as np
from boundary_loss.losses import BoundaryLoss
from boundary_loss.dataloader import dist_map_transform
from ray import tune
import ray
from ray.train import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch

def distance_transform(binary_image, iterations=20):
    """
    Approximate distance transform of a binary image tensor using dilations.
    The result is an image where the intensity is inversely proportional to the distance from the nearest boundary.
    
    Args:
    binary_image (torch.Tensor): A binary image tensor of shape (H, W).
    iterations (int): The number of iterations to perform the dilation, which affects the smoothness of the distance map.
    
    Returns:
    torch.Tensor: The approximated distance transform of the binary image.
    """
    # Invert the binary image since we're interested in the distance to the boundary
    bin_img = binary_image.float().unsqueeze(0)
    inverted_image = 1 - binary_image.float().unsqueeze(0)
    
    # Prepare a kernel for dilation
    kernel_size = 3
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=inverted_image.device)
    
    # Initialize distance map
    distance_map = torch.zeros_like(bin_img)
    distance_map_inv = torch.zeros_like(inverted_image)
    
    # Iterate to accumulate the distance map
    for i in range(1, iterations + 1):
        # Dilation step
        dilated_image = F.conv2d(bin_img, kernel, padding=1)
        dilated_image_inv = F.conv2d(inverted_image, kernel, padding=1)
        
        # Update the distance map where the dilation has reached new pixels
        distance_map += (dilated_image > 0).float() #* (distance_map == 0).float()
        distance_map_inv += (dilated_image_inv > 0).float()
        
        # Update the binary image for the next iteration
        bin_img = (dilated_image > 0).float()
        inverted_image = (dilated_image_inv > 0).float()
    
    # Normalize the distance map
    distance_map /= distance_map.max()
        
    # Normalize the distance map inv
    distance_map_inv /= distance_map_inv.max()
    
    # combine distance maps
    distance_map[distance_map == 1.0] = 0.0
    distance_map_inv[distance_map_inv == 1.0] = 0.0
    distance_map += distance_map_inv
    distance_map /= distance_map.max()
    
    
    
    return distance_map

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, hparams):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    mask_loss_func = nn.BCELoss()
    gaussian_blur = GaussianBlur(11)
    disttransform = dist_map_transform([1, 1], 2)
    # boundary_loss_func = BoundaryLoss(idc=[1])
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
        # gt_dist_map = disttransform(gt_mask_bin.cpu()).to(gt_mask_bin.device)
        
        # gt_depth_view = viewpoint_cam.original_depth_mask.cuda()
        
        # gt_mask_blurred = torch.unsqueeze(gt_mask, 0)
        # gt_mask_blurred = gt_mask_blurred.repeat(1, 3, 1, 1)
        
        # blur the ground truth
        # gt_mask_blurred = F.interpolate(gt_mask_blurred, scale_factor=1/10, mode='bilinear', align_corners=False)
        # gt_mask_blurred = F.interpolate(gt_mask_blurred, scale_factor=10, mode='bilinear', align_corners=False)
        # gt_mask_blurred = gaussian_blur(gt_mask_blurred)
        # gt_mask_blurred.squeeze_(0)
        # gt_mask_blurred = rgb_to_grayscale(gt_mask_blurred)
        
        boundary_loss_raw = boundary_loss_func(mask, gt_mask_bin, hparams['boundary_dice_prop'], hparams['boundary_bce_prop'])
        
        Ll1_image = l1_loss(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * Ll1_image + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        annealed_image_loss = sigmoid_anneal(loss_image, iteration, max_iterations, hparams['img_sigmoid_p'], hparams['img_sigmoid_k'])

        Ll1_depth = l1_loss(depth, gt_mask)        
        ssim_depth = 1.0 - ssim(depth, gt_mask)
        
        # loss_depth = (1.0 - opt.lambda_dssim) * Ll1_depth + opt.lambda_dssim * (1.0 - ssim(depth, gt_mask))
        loss_depth = weighted_loss_sum([hparams['Ll1_depth'], hparams['ssim_depth'], hparams['boundary_depth']], [Ll1_depth, ssim_depth, boundary_loss_raw])
        annealed_depth_loss = sigmoid_anneal(loss_depth, iteration, max_iterations, hparams['depth_sigmoid_p'], hparams['depth_sigmoid_k'])
        # cos_anneal_loss(loss_depth, iteration, max_iterations)
        # loss = weighted_loss_sum(1, 1, loss_image, annealed_mask_loss)
        
        # boundary_loss_raw = boundary_loss_func(mask_bin, gt_dist_map)
        # boundary_loss = boundary_loss_raw
        # boundary_loss =  sin_grow_loss(boundary_loss_raw, iteration, max_iterations)
        # loss = weighted_loss_sum(1, 2, loss, boundary_loss)
        
        scaling_loss = gaussians.scaling_reg_loss()
        annealed_scaling_loss = sigmoid_anneal(scaling_loss, iteration, max_iterations, hparams['depth_sigmoid_p'], hparams['depth_sigmoid_k'])
        # loss = weighted_loss_sum(1, 0.1, loss, annealed_scaling_loss)
        # num_p_loss = 1.0/gaussians.get_xyz.numel()
        # loss = weighted_loss_sum(1, 1, loss, num_p_loss)
        loss = weighted_loss_sum([hparams['image_loss'], hparams['depth_loss'], hparams['scaling_loss']], [annealed_image_loss, annealed_depth_loss, annealed_scaling_loss])
        loss *= 10
        # loss *= hparams['loss_multiplier']
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
            if iteration < opt.densify_until_iter: #len(gaussians.get_xyz) > opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
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
    result = {}
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss_image', Ll1_image.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_mask', loss_mask.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/boundary_loss_raw', boundary_loss_raw.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    gaussian_blur = GaussianBlur(11)

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
                    # mask_bin = (mask > 0).float()
                    # gt_mask_bin = (gt_mask > 0).float()
                    # blur the ground truth
                    # gt_mask_blurred = torch.unsqueeze(gt_mask, 0)
                    # gt_mask_blurred = gt_mask_blurred.repeat(1, 3, 1, 1)
                    # # gt_mask_blurred = F.interpolate(gt_mask_blurred, scale_factor=1/10, mode='bilinear', align_corners=False)
                    # # gt_mask_blurred = F.interpolate(gt_mask_blurred, scale_factor=10, mode='bilinear', align_corners=False)
                    # gt_mask_blurred = gaussian_blur(gt_mask_blurred)
                    # gt_mask_blurred.squeeze_(0)
                    # gt_mask_blurred = rgb_to_grayscale(gt_mask_blurred) 
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_mask".format(viewpoint.image_name), mask[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_depths".format(viewpoint.image_name), depths[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask".format(viewpoint.image_name), gt_mask[None], global_step=iteration)
                            # tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask_blurred".format(viewpoint.image_name), gt_mask_blurred, global_step=iteration)
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
    
    if args.hparam_tune:
        def trainable(config):
            results = training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, config)
            return results
            
        algo = BayesOptSearch()
        algo = ConcurrencyLimiter(algo, max_concurrent=4)
        num_samples = 100
        
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
            # "scale_sigmoid_p": tune.uniform(0.0, 1.1),
            # "scale_sigmoid_k": tune.uniform(-1.0, 1.0),
            'image_loss': tune.uniform(0.0, 1.0),
            'depth_loss': tune.uniform(0.0, 1.0),
            'scaling_loss': tune.uniform(0.0, 1.0),
            # 'loss_multiplier': tune.uniform(0.0, 33.0)
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
    else:
        # hparam = {'boundary_dice_prop': 0.26725613777853285, 'boundary_bce_prop': 0.36238144817574747, 'img_sigmoid_p': 0.08171925078722586, 'img_sigmoid_k': 0.5918202718904957, 'Ll1_depth': 0.45512229279602723, 'ssim_depth': 0.6778992259613144, 'boundary_depth': 0.10143219876129972, 'depth_sigmoid_p': 0.3209938517623887, 'depth_sigmoid_k': 0.40587387632931615, 'image_loss': 0.5463820596844736, 'depth_loss': 0.022054009819334253, 'scaling_loss': 0.2792196807170144}
        # hparam = {'boundary_dice_prop': 0.18340450985343382, 'boundary_bce_prop': 0.21233911067827616, 'img_sigmoid_p': 0.15344324671724602, 'img_sigmoid_k': 0.22370578944475894, 'Ll1_depth': 0.8324426408004217, 'ssim_depth': 0.3663618432936917, 'boundary_depth': 0.18182496720710062, 'depth_sigmoid_p': 0.47513952050632735, 'depth_sigmoid_k': 0.04951286326447568, 'image_loss': 0.2912291401980419, 'depth_loss': 0.3042422429595377, 'scaling_loss': 0.29214464853521815}
        # hparam = {'boundary_dice_prop': 0.4401524937396013, 'boundary_bce_prop': 0.09767211400638387, 'img_sigmoid_p': 0.7287745127893802, 'img_sigmoid_k': -0.48244003679996617, 'Ll1_depth': 0.3046137691733707, 'ssim_depth': 0.5200680211778108, 'boundary_depth': 0.6842330265121569, 'depth_sigmoid_p': 0.037827373226740235, 'depth_sigmoid_k': -0.00964617977745963, 'image_loss': 0.9093204020787821, 'depth_loss': 0.12203823484477883, 'scaling_loss': 0.31171107608941095}
        # hparam = {'boundary_dice_prop': 0.5986584841970366, 'boundary_bce_prop': 0.9507143064099162, 'img_sigmoid_p': 0.7788798355756501, 'img_sigmoid_k': 0.2022300234864176, 'Ll1_depth': 0.3745401188473625, 'ssim_depth': 0.9699098521619943, 'boundary_depth': 0.7319939418114051, 'depth_sigmoid_p': 0.06389197338501941, 'depth_sigmoid_k': -0.6880109593275947, 'image_loss': 0.8661761457749352, 'depth_loss': 0.15601864044243652, 'scaling_loss': 0.020584494295802447}
        
        # Nov 9 hparam tuning on image and depth psnrs, no boundary loss though:
        hparam = {'boundary_dice_prop': 0.033233219399004026, 'boundary_bce_prop': 0.6436413807391984, 'img_sigmoid_p': 0.01008129025382042, 'img_sigmoid_k': -0.9084480195520008, 'Ll1_depth': 0.9310024229990749, 'ssim_depth': 0.0714191045525289, 'boundary_depth': 0.05139642974053518, 'depth_sigmoid_p': 0.09626558054180495, 'depth_sigmoid_k': -0.6637714408505092, 'image_loss': 0.5443212115275131, 'depth_loss': 0.8516824065619147, 'scaling_loss': 0.1859502986539709}
        
        # Nov 10 combined metric loss
        hparams={'boundary_dice_prop': 0.7072663092769335, 'boundary_bce_prop': 0.7567179358568938, 'img_sigmoid_p': 1.086743312041734, 'img_sigmoid_k': -0.5933987285312401, 'Ll1_depth': 0.38159149238473783, 'ssim_depth': 0.1257942061432118, 'boundary_depth': 0.44228481357720983, 'depth_sigmoid_p': 0.709852751117558, 'depth_sigmoid_k': -0.07288362148599577, 'image_loss': 0.4529836507526636, 'depth_loss': 0.8461601421454328, 'scaling_loss': 0.009799058383410901}
        training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, hparam)

    # All done
    print("\nTraining complete.")
