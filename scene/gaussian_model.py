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

from typing import List
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB, C0
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class CompositeGaussianModel:
    def __init__(self, sh_degree : int, model_idxs : List[int]):
        self.max_sh_degree = sh_degree  
        self.model_count = len(model_idxs)
        self.models = {}
        for i in model_idxs:
            self.models[i] = GaussianModel(self.max_sh_degree)
        self.active_sh_degree = 0
    
    def __getitem__(self, key):
        return self.models[key]
        
    def get_scaling(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        scalings = [self.models[m].get_scaling for m in model_idxs]
        scalings = [r.unsqueeze(0) if len(r.shape) == 1 else r for r in scalings]
        return torch.cat((scalings), 0)
    
    def get_rotation(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        rotations = [self.models[m].get_rotation for m in model_idxs]
        rotations = [r.unsqueeze(0) if len(r.shape) == 1 else r for r in rotations]
        return torch.cat((rotations), 0)
    
    def get_xyz(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        xyzs = [self.models[m].get_xyz for m in model_idxs]
        xyzs = [points.unsqueeze(0) if len(points.shape) == 1 else points for points in xyzs]
        try:
            return torch.cat((xyzs), 0)
        except:
            print([v.shape for v in xyzs])
    
    def get_max_radii2D(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        max_radii2Ds = [self.models[m].max_radii2D for m in model_idxs]
        max_radii2Ds = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in max_radii2Ds]
        print("shapes", [m.shape for m in max_radii2Ds])
        return torch.cat((max_radii2Ds), 0)

    def get_xyz_gradient_accum(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        xyz_gradient_accums = [self.models[m].xyz_gradient_accum for m in model_idxs]
        xyz_gradient_accums = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in xyz_gradient_accums]
        return torch.cat((xyz_gradient_accums), 0)
    
    def get_denom(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        denoms = [self.models[m].denom for m in model_idxs]
        denoms = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in denoms]
        return torch.cat((denoms), 0)
    
    def get_features_dc(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        features_dcs = [self.models[m]._features_dc for m in model_idxs]
        features_dcs = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in features_dcs]
        return torch.cat((features_dcs), dim=0)
    
    def get_features_rest(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        features_rests = [self.models[m]._features_rest for m in model_idxs]
        features_rests = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in features_rests]
        return torch.cat((features_rests), dim=0)
    
    def get_features(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        features_dcs = self.get_features_dc(model_idxs)
        features_rests = self.get_features_rest(model_idxs)
        return torch.cat((features_dcs, features_rests), dim=1)
    
    def get_opacity(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        opacities = [self.models[m].get_opacity for m in model_idxs]
        opacities = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in opacities]
        return torch.cat((opacities), dim=0)
    
    def get_geometry_opacity(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        geometry_opacities = [self.models[m].get_geometry_opacity for m in model_idxs]
        geometry_opacities = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in geometry_opacities]
        return torch.cat((geometry_opacities), dim=0)
    
    def get_spatial_lr_scale(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        spatial_lr_scales = [self.models[m].spatial_lr_scale for m in model_idxs]
        spatial_lr_scales = [o.unsqueeze(0) if len(o.shape) == 1 else o for o in spatial_lr_scales]
        return torch.tensor(spatial_lr_scales)
    
    def scaling_reg_loss(self):
        scalings = torch.sigmoid(self.get_scaling())
        alpha = 10
        beta = 5
        return beta * torch.mean(scalings) + alpha * torch.std(scalings)
    
    def capture(self, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        return (
            self.active_sh_degree,
            self.get_xyz(model_idxs),
            self.get_features_dc(model_idxs),
            self.get_features_rest(model_idxs),
            self.get_scaling(model_idxs),
            self.get_rotation(model_idxs),
            self.get_opacity(model_idxs),
            self.get_geometry_opacity(model_idxs),
            self.get_max_radii2D(model_idxs),
            self.get_xyz_gradient_accum(model_idxs),
            self.get_denom(model_idxs),
            # self.optimizer.state_dict(),
            self.get_spatial_lr_scale(model_idxs),
        )
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('geometry_opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, model_path, load_iter):
        model_path = os.path.dirname(model_path)
        for i in self.models.keys():
            subpath = str(i).zfill(3)
            load_path = os.path.join(model_path, subpath,
                            "point_cloud",
                            "iteration_" + str(load_iter),
                            "point_cloud.ply")
            self.models[i].save_ply(load_path)
            assert(os.path.exists(load_path))
            
        # path_dir = os.path.dirname(path)
        # mkdir_p(path_dir)

        # all_model_idxs = [i for i in range(self.model_count)]
        # xyz = self.get_xyz(all_model_idxs).detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        # f_dc = self.get_features_dc(all_model_idxs).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self.get_features_rest(all_model_idxs).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self.get_opacity(all_model_idxs).detach().cpu().numpy()
        # geometry_opacities = self.get_geometry_opacity(all_model_idxs).detach().cpu().numpy()
        # scale = self.get_scaling(all_model_idxs).detach().cpu().numpy()
        # rotation = self.get_rotation(all_model_idxs).detach().cpu().numpy()

        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, geometry_opacities,
        #                                 scale, rotation), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # el = PlyElement.describe(elements, 'vertex')
        # PlyData([el]).write(path)
                    
    def create_from_pcd(self, pcds, spatial_lr_scales, model_idxs=None):
        model_idxs = list(self.models.keys()) if model_idxs is None else model_idxs
        for m in model_idxs:
            self.models[m].create_from_pcd(pcds[m], spatial_lr_scales[m])
            
    def restore(self, model_args, training_args):
        for m in self.models.keys():
            self.models[m].restore(model_args, training_args)
    
    def training_setup(self, training_args):
        for m in self.models.keys():
            self.models[m].training_setup(training_args)
        
    def update_learning_rate(self, iteration):
        lrs = [self.models[m].update_learning_rate(iteration) for m in self.models.keys()]
        return lrs

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def load_ply(self, model_path, load_iter):
        model_path = os.path.dirname(model_path)
        for i in self.models.keys():
            subpath = str(i).zfill(3)
            load_path = os.path.join(model_path, subpath,
                            "point_cloud",
                            "iteration_" + str(load_iter),
                            "point_cloud.ply")
            self.models[i].load_ply(load_path)
            
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._geometry_opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._geometry_opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._geometry_opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        try:
            return self.rotation_activation(self._rotation)
        except:
            return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        try:
            res = torch.cat((features_dc, features_rest), dim=1)
        except:
            res = None
            print("issue with getting features")
            print(features_dc.shape, features_rest.shape)
        return res
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)        
    
    @property
    def get_geometry_opacity(self):
        return self.opacity_activation(self._geometry_opacity)
    
    def scaling_reg_loss(self):
        scalings = torch.sigmoid(self._scaling)
        alpha = 10
        beta = 5
        return beta * torch.mean(scalings) + alpha * torch.std(scalings)
    
    def black_spot_loss(self):
        black_opaques = torch.exp(10 * (self.get_opacity -1)) * (1 - (torch.mean(SH2RGB(self._features_dc), dim=-1)))
        black_opaques = black_opaques.squeeze(-1)
        return torch.mean(torch.topk(black_opaques, k=int(black_opaques.shape[0])//20).values)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        geometry_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._geometry_opacity = nn.Parameter(geometry_opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._geometry_opacity], 'lr': training_args.geometry_opacity_lr, "name": "geometry_opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('geometry_opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        geometry_opacities = self._geometry_opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, geometry_opacities,
                                        scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities,
        #                                 scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        geometry_opacities_new = inverse_sigmoid(torch.min(self.get_geometry_opacity, torch.ones_like(self.get_geometry_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(geometry_opacities_new, "geometry_opacity")
        self._geometry_opacity = optimizable_tensors["geometry_opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        geometry_opacities = np.asarray(plydata.elements[0]["geometry_opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._geometry_opacity = nn.Parameter(torch.tensor(geometry_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._geometry_opacity = optimizable_tensors["geometry_opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_geometry_opacities, 
                                new_scaling, new_rotation):
    # def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, 
    #                             new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "geometry_opacity": new_geometry_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._geometry_opacity = optimizable_tensors["geometry_opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_geometry_opacity = self._geometry_opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_geometry_opacity,
                                   new_scaling, new_rotation)
        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,
        #                            new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_geometry_opacities = self._geometry_opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_geometry_opacities,
                                   new_scaling, new_rotation)
        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,
        #                            new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, min_geometry_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if len(self.get_xyz) <= 100_000:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        # min_opacity = min(min_opacity, self.get_opacity.max())
        if self.get_opacity.shape[0] == 1:
            prune_mask = torch.ones((1)).bool()
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        # if len(prune_mask.shape) != 1:
        #     prune_mask = torch.ones_like(self.get_opacity) > 0
        assert len(prune_mask.shape) == 1, f"prune mask shape {prune_mask.shape} {self.get_opacity.shape}"
        # prune_geometry_mask = (self.get_geometry_opacity < min_geometry_opacity).squeeze()
        # prune_mask = torch.logical_or(prune_mask, prune_geometry_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
        if torch.all(prune_mask):
            # assert False, "Tried to remove all points"
            # print("Tried to remove all points")
            prune_mask = False
            
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, all_visible=False):
        if all_visible:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1        
        