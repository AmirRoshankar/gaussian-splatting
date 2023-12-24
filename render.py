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

import shutil
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn.functional as F
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, overwrite):
    '''
        Render images and masks for evaluation
        
        :param model_path: Path to the model
        :param name: Name of the model
        :param iteration: Iteration of the model to render
        :param views: Views to render from
        :param gaussians: Gaussian model to render
        :param pipeline: Pipeline for rendering
        :param background: Background colour for rendering
        :param overwrite: Whether to overwrite the existing renders   
    '''
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_masks")
    gt_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_mask")
    gt_masks_blur_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_mask_blur")

    if overwrite:
        if os.path.exists(render_path):
            shutil.rmtree(render_path)
        if os.path.exists(gts_path):
            shutil.rmtree(gts_path)
        if os.path.exists(render_mask_path):
            shutil.rmtree(render_mask_path)
        if os.path.exists(gt_masks_path):
            shutil.rmtree(gt_masks_path)
        if os.path.exists(gt_masks_blur_path):
            shutil.rmtree(gt_masks_blur_path)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_mask_path, exist_ok=True)
    makedirs(gt_masks_path, exist_ok=True)
    makedirs(gt_masks_blur_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        model_renders = render(view, gaussians, pipeline, background)
        rendering = model_renders["render"]
        mask = model_renders["mask"]
        gt = view.original_image[0:3, :, :]
        gt_mask = view.original_mask
        
        # blur the ground truth
        gt_mask_blurred = torch.unsqueeze(gt_mask, 0)
        gt_mask_blurred = gt_mask_blurred.repeat(1, 3, 1, 1)
        gt_mask_blurred = F.interpolate(gt_mask_blurred, scale_factor=1/10, mode='bilinear', align_corners=False)
        gt_mask_blurred = F.interpolate(gt_mask_blurred, scale_factor=10, mode='bilinear', align_corners=False)
        gt_mask_blurred.squeeze_(0)
        gt_mask_blurred = rgb_to_grayscale(gt_mask_blurred) 
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(mask, os.path.join(render_mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_mask, os.path.join(gt_masks_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_mask_blurred, os.path.join(gt_masks_blur_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, overwrite : bool):
    '''
        Render train and test sets
        
        :param dataset: Dataset for Gaussian model initialization
        :param iteration: Iteration of the model to render
        :param pipeline: Pipeline for rendering
        :param skip_train: Whether to skip the training set
        :param skip_test: Whether to skip the testing set
        :param overwrite: Whether to overwrite the existing renders   
    '''
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, overwrite)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, overwrite)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.overwrite)