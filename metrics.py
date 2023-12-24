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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, dice_loss
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, render_masks_dir, gt_mask_dir):
    '''
        Load in images and masks for renders and ground truths
        
        :param renders_dir: Path of rendered images
        :param gt_dir: Path to ground truth images
        :param renders_dir: Path of rendered masks
        :param gt_dir: Path to ground truth masks
        
        :returns: rendered images, ground truth images, rendered masks, ground truth masks, image names
    '''
    renders = []
    gts = []
    masks = []
    gt_masks = []
    # gt_masks_blur = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(render_masks_dir / fname)
        gt_mask = Image.open(gt_mask_dir / fname)
        # gt_mask_blur = Image.open(gt_mask_blur_dir / fname)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :, :, :].cuda())
        gt_masks.append(tf.to_tensor(gt_mask).unsqueeze(0)[:, :, :, :].cuda())
        # gt_masks_blur.append(tf.to_tensor(gt_mask_blur).unsqueeze(0)[:, :, :, :].cuda())
        image_names.append(fname)
    return renders, gts, masks, gt_masks, image_names

def evaluate(model_paths):
    '''
        Calculate performance metrics for the model at the given path
        
        :param model_paths: List of model paths
    '''
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                gt_mask_dir = method_dir/ "gt_mask"
                # gt_mask_blur_dir = method_dir/ "gt_mask_blur"
                renders_dir = method_dir / "renders"
                render_masks_dir = method_dir / "render_masks"
                renders, gts, masks, gt_masks, image_names = readImages(renders_dir, gt_dir, render_masks_dir, gt_mask_dir)

                # image metrics
                image_ssims = []
                image_psnrs = []
                image_lpipss = []

                # mask metrics
                mask_ssims = []           
                mask_psnrs = []      
                mask_lpipss = [] 
                mask_dices = []
                
                # metrics for blurred masks
                # mask_blur_ssims = []           
                # mask_blur_psnrs = []      
                # mask_blur_lpipss = [] 
                # mask_blur_dices = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    image_ssims.append(ssim(renders[idx], gts[idx]))
                    image_psnrs.append(psnr(renders[idx], gts[idx]))
                    image_lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    mask_ssims.append(ssim(masks[idx], gt_masks[idx]))
                    mask_psnrs.append(psnr(masks[idx], gt_masks[idx]))
                    mask_lpipss.append(lpips(masks[idx], gt_masks[idx], net_type='vgg'))
                    mask_dices.append(dice_loss(masks[idx], gt_masks[idx]))

                    # mask_blur_ssims.append(ssim(masks[idx], gt_masks_blur[idx]))
                    # mask_blur_psnrs.append(psnr(masks[idx], gt_masks_blur[idx]))
                    # mask_blur_lpipss.append(lpips(masks[idx], gt_masks_blur[idx], net_type='vgg'))
                    # mask_blur_dices.append(dice_loss(masks[idx], gt_masks_blur[idx]))

                print(" Image SSIM : {:>12.7f}".format(torch.tensor(image_ssims).mean(), ".5"))
                print(" Image PSNR : {:>12.7f}".format(torch.tensor(image_psnrs).mean(), ".5"))
                print(" Image LPIPS: {:>12.7f}".format(torch.tensor(image_lpipss).mean(), ".5"))
                print("")
                print(" Mask SSIM : {:>12.7f}".format(torch.tensor(mask_ssims).mean(), ".5"))
                print(" Mask PSNR : {:>12.7f}".format(torch.tensor(mask_psnrs).mean(), ".5"))
                print(" Mask LPIPS: {:>12.7f}".format(torch.tensor(mask_lpipss).mean(), ".5"))
                print(" Mask DICE: {:>12.7f}".format(torch.tensor(mask_dices).mean(), ".5"))
                print("")
                # print(" Blur Mask SSIM : {:>12.7f}".format(torch.tensor(mask_blur_ssims).mean(), ".5"))
                # print(" Blur Mask PSNR : {:>12.7f}".format(torch.tensor(mask_blur_psnrs).mean(), ".5"))
                # print(" Blur Mask LPIPS: {:>12.7f}".format(torch.tensor(mask_blur_lpipss).mean(), ".5"))
                # print(" Blur Mask DICE: {:>12.7f}".format(torch.tensor(mask_blur_dices).mean(), ".5"))
                # print("")

                # Update each metric
                full_dict[scene_dir][method].update({"IMAGE_SSIM": torch.tensor(image_ssims).mean().item(),
                                                        "IMAGE_PSNR": torch.tensor(image_psnrs).mean().item(),
                                                        "IMAGE_LPIPS": torch.tensor(image_lpipss).mean().item(),
                                                        
                                                        "MASK_SSIM": torch.tensor(mask_ssims).mean().item(),
                                                        "MASK_PSNR": torch.tensor(mask_psnrs).mean().item(),
                                                        "MASK_LPIPS": torch.tensor(mask_lpipss).mean().item(),
                                                        "MASK_DICE": torch.tensor(mask_dices).mean().item(),})
                                                        
                                                        # "BLUR MASK_SSIM": torch.tensor(mask_blur_ssims).mean().item(),
                                                        # "BLUR_MASK_PSNR": torch.tensor(mask_blur_psnrs).mean().item(),
                                                        # "BLUR_MASK_LPIPS": torch.tensor(mask_blur_lpipss).mean().item(),
                                                        # "BLUR_MASK_DICE": torch.tensor(mask_blur_dices).mean().item()})
                
                per_view_dict[scene_dir][method].update({"IMAGE_SSIM": {name: ssim for ssim, name in zip(torch.tensor(image_ssims).tolist(), image_names)},
                                                            "IMAGE_PSNR": {name: psnr for psnr, name in zip(torch.tensor(image_psnrs).tolist(), image_names)},
                                                            "IMAGE_LPIPS": {name: lp for lp, name in zip(torch.tensor(image_lpipss).tolist(), image_names)},
                                                            
                                                            "MASK_SSIM": {name: ssim for ssim, name in zip(torch.tensor(mask_ssims).tolist(), image_names)},
                                                            "MASK_PSNR": {name: psnr for psnr, name in zip(torch.tensor(mask_psnrs).tolist(), image_names)},
                                                            "MASK_LPIPS": {name: lp for lp, name in zip(torch.tensor(mask_lpipss).tolist(), image_names)},
                                                            "MASK_DICE": {name: lp for lp, name in zip(torch.tensor(mask_dices).tolist(), image_names)},})
                                                            
                                                            # "BLUR MASK_SSIM": {name: ssim for ssim, name in zip(torch.tensor(mask_blur_ssims).tolist(), image_names)},
                                                            # "BLUR_MASK_PSNR": {name: psnr for psnr, name in zip(torch.tensor(mask_blur_psnrs).tolist(), image_names)},
                                                            # "BLUR_MASK_LPIPS": {name: lp for lp, name in zip(torch.tensor(mask_blur_lpipss).tolist(), image_names)},
                                                            # "BLUR_MASK_DICE": {name: lp for lp, name in zip(torch.tensor(mask_blur_dices).tolist(), image_names)}})

            # Save results
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("Exception occured:", e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
