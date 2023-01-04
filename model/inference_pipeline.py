import math
from random import sample
import numpy as np
import sys
import torch
import torch.nn as nn
from dataset.warp_uv_dataset import WarpUVDataset
sys.path.append('..')
from model.texture import Texture
from model.unet import UNet_Mask
from model.inference_neuralwarp import NeuralWarp_Mask 


class InferencePipeLine(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True, \
                low_pass=False, use_upconv=False, low_pass_sigma=None, device=None, \
                use_bg=None, use_warp=False, use_instance_norm=False, \
                network_size=1, predict_mask=False, cache_layer=4, concat_uv_diff=False, \
                single_temporal=False, use_pose=False, warp_sample=True, cache_features=False,
                sh_skips=False, sh_pose=False, use_mlp=True, skip_cache_dim=9, timing=False, start_recorder=None, end_recorder=None, use_exp=False,
                use_cache_layers=[3,4,5]):

        """
        NOTES: NOT using view direction is deprecated and probably won't work! 
        """

        super(InferencePipeLine, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        self.texture = Texture(W, H, feature_num, use_pyramid)
        self.use_lowpass = low_pass
        self.device = device
        self.use_upconv = use_upconv
        self.low_pass_sigma = low_pass_sigma
        self.use_bg = use_bg
        self.use_instance_norm = use_instance_norm
        self.network_size = network_size
        self.predict_mask = predict_mask
        self.concat_uv_diff = concat_uv_diff
        self.cache_layer = cache_layer
        self.single_temporal = single_temporal
        self.use_cache_layers = use_cache_layers

        self.use_pose = use_pose
        self.use_exp = use_exp
        self.mlp_dim = 32

        self.warp_sample = warp_sample
        self.cache_features = cache_features
        self.sh_skips = sh_skips
        self.skip_cache_dim = skip_cache_dim

        self.sh_pose = sh_pose
        if self.use_pose:
            if self.sh_pose:
                if self.use_exp:
                    self.mlp = nn.Sequential(
                        nn.Linear(25+50*2, self.mlp_dim),
                        nn.ReLU(inplace=True)
                    )
                else:
                    self.mlp = nn.Sequential(
                        nn.Linear(25, self.mlp_dim),
                        nn.ReLU(inplace=True)
                    )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(10, self.mlp_dim),
                    nn.ReLU(inplace=True)
                )
                        
        self.use_mlp = use_mlp
        if not self.use_mlp:
            if self.sh_pose:
                self.mlp_dim = 25
            else:
                self.mlp_dim = 10

        self.timing = timing
        self.start_recorder = start_recorder
        self.end_recorder = end_recorder

        assert self.cache_layer == 3 or self.cache_layer == 4
        
        if self.predict_mask:
            self.unet = UNet_Mask(feature_num + feature_num * self.single_temporal, 4, use_lowpass=self.use_lowpass, 
                                device=self.device, use_upconv=self.use_upconv, sigma=low_pass_sigma, use_bg=0, 
                                use_warp=use_warp, use_instance_norm=use_instance_norm, 
                                network_size=self.network_size, cache_layer=self.cache_layer, cache_features=self.cache_features, skip_cache_dim=self.skip_cache_dim,
                                use_cache_layers=self.use_cache_layers)
        if self.predict_mask:
            self.warpnet = NeuralWarp_Mask(self.unet.cache_dim, 4, cache_layer=self.cache_layer,
                                            concat_uv_diff=self.concat_uv_diff, use_pose=self.use_pose, use_pose_mlp_dim=self.mlp_dim, 
                                            warp_sample=self.warp_sample, cache_features=self.cache_features, 
                                            sh_skips=self.sh_skips, skip_cache_dim=self.skip_cache_dim)

        # CACHED PARAMETERS
        self.inference_skip = None
        self.inference_features = None
        self.pose = None
        self.extrinsics = None
        self.uv_map = None
        self.exp = None
    
    def _spherical_harmonics_basis(self, extrinsics):
        '''
        extrinsics: a tensor shaped (N, 3) -> [x,y,z]
        output: a tensor shaped (N, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float, device=extrinsics.device)
        coff_0 = 1 / (2.0*math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]) * coff_2
        return sh_bands

    def parse_inputs(self, dict):
        if self.use_exp:
            return [dict['uv_map'], dict['extrinsics'], 1., dict['pose'], dict['exp']]
        else:
            return [dict['uv_map'], dict['extrinsics'], 1., dict['pose']]

    def update_cache(self, input):
        if self.use_exp:
            uv_map, extrinsics, scale, pose, exp = self.parse_inputs(input)
        else:
            uv_map, extrinsics, scale, pose = self.parse_inputs(input)
        x = self.texture(uv_map)
        basis = self._spherical_harmonics_basis(extrinsics)
        basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
        x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis

        _, cache, skip_connection = self.unet(x)

        # CACHING
        self.inference_features = cache
        self.inference_skip = [skip_connection[i] for i in range(len(skip_connection))]
        self.extrinsics = extrinsics
        self.pose = pose
        self.uv_map = uv_map
        if self.use_exp:
            self.exp = exp

    def forward(self, input):
        if self.use_exp:
            uv_map, extrinsics, scale, pose, exp = self.parse_inputs(input)
        else:
            uv_map, extrinsics, scale, pose = self.parse_inputs(input)

        if self.use_pose:
            if self.sh_pose:
                skip_basis_2 = self._spherical_harmonics_basis(pose[:,:3].clone())
                if self.use_exp:
                    mlp_input =  torch.cat((skip_basis_2.clone(), pose[:, ...].clone(), (pose[:, ...]-self.pose).clone() * 1., (extrinsics[:,:2]-self.extrinsics[:,:2]).clone()*1., extrinsics[:,:2].clone(), exp[:,...].clone(), (exp[:,...]-self.exp).clone()*1.), dim=-1)
                else:
                    mlp_input =  torch.cat((skip_basis_2.clone(), pose[:, ...].clone(), (pose[:, ...]-self.pose).clone() * 1., (extrinsics[:,:2]-self.extrinsics[:,:2]).clone()*1., extrinsics[:,:2].clone()), dim=-1)
            
            if self.use_mlp:
                mlp_out = self.mlp(mlp_input)

                mlp_out = mlp_out[..., None, None].repeat(1,1,self.inference_features.shape[-2], self.inference_features.shape[-1])
            cache = torch.cat((self.inference_features.clone(), mlp_out), dim=1)

        y2, _ = self.warpnet(cache.clone(), uv_map.clone(), self.uv_map.clone(), extrinsics, [self.inference_skip[i].clone() for i in range(2)], self.texture, pose, self.pose.clone())
        
        if self.predict_mask:
            pred_rgb_2 = y2[:, 0:3, ...]
            pred_mask_2 = torch.sigmoid(y2[:, 3, ...]).unsqueeze(1)

            return pred_rgb_2, pred_mask_2
        else: 
            return y2



        