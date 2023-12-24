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

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from math import exp

bce_loss = nn.BCELoss()

# Image dilating function for torch
def dilate(input_tensor, kernel_size):
    # Create a kernel with ones of size kernel_size
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=input_tensor.device)
    
    # Use the dilation operation
    dilated_output = F.conv2d(input_tensor.unsqueeze(0), kernel, padding=(kernel_size - 1) // 2)
    
    # Remove the extra dimensions added by unsqueeze
    dilated_output = dilated_output.squeeze(0)
    dilated_output = dilated_output > 0
    
    return dilated_output.int()

# Image eroding function for torch
def erode(input_tensor, kernel_size):
    # Create a kernel with ones of size kernel_size
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=input_tensor.device)
    
    # Use the erosion operation
    eroded_output = 1 - F.conv2d(1 - input_tensor.unsqueeze(0), kernel, padding=(kernel_size - 1) // 2)
    
    # Remove the extra dimensions added by unsqueeze
    eroded_output = eroded_output.squeeze(0)
    eroded_output = eroded_output > 0
    
    return eroded_output.int()

def sigmoid_anneal(loss, it, max_its, prop, k=1.0):
    '''
        Annealing function for smoothly turning a loss term on/off
        :param loss: Given raw loss value
        :param it: Current iteration in training
        :param max_its: Maximum number of iterations in training
        :param prop: Proportion of max iterations before inflection
        :param k: Shape of inflection
    '''
    x = torch.ones((1)).to(loss.device) * it
    x = (x - max_its * prop) / (k * max_its)
    return loss * torch.sigmoid(x)

# Cosine annealing loss
def cos_anneal_loss(loss, it, max_its):
    return loss * math.cos(it * math.pi / (2 * max_its))

# Weighted average of losses
def weighted_loss_sum(weights, losses):
    num = sum([w * l for w,l in zip(weights, losses)])
    denom = sum(weights)
    return num / denom

# IOU loss implementation
def dice_loss(prediction, target):
    intersection = torch.sum(target * prediction)
    union = torch.sum(target + prediction)
    return 1.0 - 2 * intersection / union

# Boundary loss calculated by masking boundary of mask and calculting
#   dice loss on just that region
def boundary_loss_func(prediction, target, dice_prop, bce_prop):
    erode_target = erode(target, 13)
    dilate_target = dilate(target, 21)
    mask = dilate_target - erode_target
    masked_pred = mask * prediction
    masked_target = mask * target
    
    dice = dice_loss(masked_pred, masked_target)
    bce = bce_loss(masked_pred, masked_target)
    
    return weighted_loss_sum([dice_prop, bce_prop], [dice, bce])

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

