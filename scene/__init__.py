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

import copy
import os
import random
import json
import numpy as np
from typing import List
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, CompositeGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class CompositeScene:    
    def __init__(self, args : ModelParams, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        Implementation of the scene class for a composite layered subject
        
        :param path: Path to colmap scene main folder.
        """
        self.num_scenes = len(os.listdir(args.source_path)) - 1
        self.model_path = args.model_path
        self.max_sh_degree = args.sh_degree

        self.args_list = {}
        self.load_iteration = load_iteration
        self.shuffle = shuffle
        self.resolution_scales = resolution_scales
        
        self.args = args
    
    def set_combined_scene(self, model_idxs, iteration):
        # Initialize the combined gausssian object
        self.combined_gaussian = CompositeGaussianModel(self.max_sh_degree, model_idxs=model_idxs)
        
        self.combined_args = copy.deepcopy(self.args)
        self.combined_args.source_path += "/composite"
        self.combined_args.model_path += "/composite"
        os.makedirs(self.combined_args.model_path, exist_ok=True)
        self.combined_scene = Scene(self.combined_args, 
                             self.combined_gaussian, 
                             iteration, 
                             self.shuffle, 
                             self.resolution_scales,
                             combined=True)
    
    # Initialize a the current scene for a given layer
    def set_cur_scene(self, i):
        self.args_list[i] = copy.deepcopy(self.args)
        self.args_list[i].source_path += "/" + str(i).zfill(3)
        self.args_list[i].model_path += "/" + str(i).zfill(3)
        os.makedirs(self.args_list[i].model_path, exist_ok=True)

        self.cur_gaussian = GaussianModel(self.max_sh_degree)
        self.cur_scene = Scene(self.args_list[i], 
                             self.cur_gaussian, 
                             self.load_iteration, 
                             self.shuffle, 
                             self.resolution_scales)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], combined=False):
        """
        Implementation of a scene object for linking ground truth views with a Gaussian model
        
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type! " + str(os.path.join(args.source_path, "transforms_train.json"))

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if combined:
                self.gaussians.load_ply(self.model_path, self.loaded_iter)
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, combined=False):
        if combined:
            self.gaussians.save_ply(self.model_path, iteration)
        else:
            iteration_path = f"point_cloud/iteration_{iteration}"
            point_cloud_path = os.path.join(self.model_path, iteration_path)
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]