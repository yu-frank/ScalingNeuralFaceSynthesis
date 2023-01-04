import math
from random import sample

import numpy as np
import sys
import torch
import torch.nn as nn

sys.path.append('..')
from model.texture import Texture
from model.unet import UNet_Mask
from model.neuralwarp import NeuralWarp_Mask

import torch.nn.functional as F


class PipeLine(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True, \
                low_pass=True, use_upconv=True, low_pass_sigma=2, device='cuda', \
                use_warp=True, use_instance_norm=False, \
                network_size=2, predict_mask=True, cache_layer=3, concat_uv_diff=True, \
                single_temporal=False, use_pose=True, warp_sample=False, cache_features=True,
                sh_skips=False, sh_pose=False, use_mlp=True, skip_cache_dim=8, timing=False, 
                start_recorder=None, end_recorder=None, use_exp=True, use_cache_layers=[3,4,5]):

        """
        W (int): Neural texture width
        H (int): Neural texture height
        feature_num (int): Number of features in the neural texture
        use_pyramid (bool): Use Laplacian Pyramid for the neural texture (from DNR) (default: True)
        view_direction (bool): condition on DECA camera extrinsics (default: True)
        low_pass (bool): use a low-pass filter on UNet features (default: True)
        use_upconv (bool): use up-convolutions or transposed convolutions (default: True)
        low_pass_sigma (float): sigma value for low pass filter (if used) (default: 2.0)
        device (string): which device to load model onto ('cpu' or 'cuda') (default: 'cuda')
        use_warp (bool): implicitly warp from cached values of t to t+t_diff frame
        use_instance_norm (bool): use instance norm in the UNet (generator) (default: False)
        network_size (int): multiplicitive size for the UNet (generator) (default: 2)
        predict_mask (bool): have the generator and warp network predict the foreground mask (default: True)
        cache_layer (int): which layer to cache information from the generator (default: 3)
        concact_uv_diff (bool): concatenate the difference between t and t+t_diff
        UV maps in warp network (default: True)
        single_temporal (bool): ONLY used for non-warping baseline (default: False)
        use_pose (bool): condition on the FLAME pose parameters (default: True)
        warp_sample (bool): whether or not to explicitly sample UV map in the warp network (default: False)
        cache_features (bool): whether or not we are caching features (default: True)
        sh_skips (bool): apply spherical harmonics to the cached skip-connection features (default: True)
        sh_pose (bool): apply spherical harmonics to the t+t_diff FlAME pose parameters (default: True)
        use_mlp (bool): pass new and cached FLAME parameters through an MLP (default: True)
        skip_cache_dim (int): dimension of the skip-connection parameters that are cached (default: 8)
        timing (bool): whether or not to time network components (default: False)
        start_recorder (torch.cuda.Event): used for timing network components (default: None)
        end_recorder (torch.cuda.Event): used for timing network components (default: None)
        use_exp (bool): condition network on FLAME expression parameters (default: True)
        use_cache_layers (list[int]): which layers to cache (generator) (default: [3,4,5])
        """

        super(PipeLine, self).__init__()
        assert use_warp != single_temporal

        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        self.texture = Texture(W, H, feature_num, use_pyramid)
        self.use_lowpass = low_pass
        self.device = device
        self.use_upconv = use_upconv
        self.low_pass_sigma = low_pass_sigma
        self.use_instance_norm = use_instance_norm
        self.network_size = network_size
        self.predict_mask = predict_mask
        self.concat_uv_diff = concat_uv_diff
        self.cache_layer = cache_layer
        self.single_temporal = single_temporal

        self.inference_skip = None
        self.inference_features = None

        self.use_pose = use_pose
        self.mlp_dim = 32

        self.warp_sample = warp_sample
        self.cache_features = cache_features
        self.sh_skips = sh_skips
        self.skip_cache_dim = skip_cache_dim

        self.use_exp = use_exp

        self.timing = timing
        if self.timing:
            self.start_recorder = start_recorder
            self.end_recorder = end_recorder

        self.use_cache_layers = use_cache_layers

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

        assert self.cache_layer == 3 or self.cache_layer == 4
        
        self.unet = UNet_Mask(feature_num + feature_num * self.single_temporal, 4, use_lowpass=self.use_lowpass, 
                            device=self.device, use_upconv=self.use_upconv, sigma=low_pass_sigma,
                            use_warp=use_warp, use_instance_norm=use_instance_norm, 
                            network_size=self.network_size, cache_layer=self.cache_layer, cache_features=self.cache_features, skip_cache_dim=self.skip_cache_dim,
                            use_cache_layers=self.use_cache_layers)
     

        self.use_warp = use_warp   
        if self.use_warp:
            if self.predict_mask:
                self.warpnet = NeuralWarp_Mask(self.unet.cache_dim, 4, cache_layer=self.cache_layer,
                                                concat_uv_diff=self.concat_uv_diff, use_pose=self.use_pose, use_pose_mlp_dim=self.mlp_dim, 
                                                warp_sample=self.warp_sample, cache_features=self.cache_features, 
                                                sh_skips=self.sh_skips, skip_cache_dim=self.skip_cache_dim)

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

    def forward(self, *args):
        # BASELINE
        if not self.use_warp: 
            if self.timing:
                self.start_recorder.record()
            if self.view_direction:
                uv_map, extrinsics, device, scale, bg = args
                if self.single_temporal:
                    x = self.texture(uv_map[:,0,...])
                    x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                    assert x.shape[1] >= 12
                    basis = self._spherical_harmonics_basis(extrinsics[:,0,...])
                    basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
                    x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis

                    x1 = self.texture(uv_map[:,1,...])
                    x1 = F.interpolate(x1, scale_factor=scale, mode='bilinear', align_corners=False)
                    assert x1.shape[1] >= 12
                    basis1 = self._spherical_harmonics_basis(extrinsics[:,1,...])
                    basis1 = basis1.view(basis1.shape[0], basis1.shape[1], 1, 1)
                    x1[:, 3:12, :, :] = x1[:, 3:12, :, :] * basis1
                    
                    x = torch.cat((x,x1), dim=1)
                else:
                    x = self.texture(uv_map)
                    x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                    assert x.shape[1] >= 12
                    basis = self._spherical_harmonics_basis(extrinsics)
                    basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
                    x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis
                    
            if self.timing: 
                self.end_recorder.record()  
                torch.cuda.synchronize()
                t1_gridsample = self.start_recorder.elapsed_time(self.end_recorder)

            if self.timing:
                self.start_recorder.record()
            y = self.unet(x)
            
            if self.timing: 
                self.end_recorder.record()  
                torch.cuda.synchronize()
                t1_generator = self.start_recorder.elapsed_time(self.end_recorder)

            if not self.predict_mask:
                return x[:, 0:3, :, :], y
            else:
                pred_rgb = y[:, 0:3, ...]
                pred_mask = torch.sigmoid(y[:, 3, ...])
                if self.timing:
                    return x[:, 0:3, :, :], pred_rgb, pred_mask.unsqueeze(1), [t1_gridsample, t1_generator] # texture, rgb output, predicted mask
                else:
                    return x[:, 0:3, :, :], pred_rgb, pred_mask.unsqueeze(1) # texture, rgb output, predicted mask

        # IMPLICIT WARPING
        else: 
            if self.view_direction:
                if self.use_pose:
                    if self.use_exp:
                        uv_map, extrinsics, device, scale, bg, pose, exp = args
                    else:
                        uv_map, extrinsics, device, scale, bg, pose = args
                else:
                    uv_map, extrinsics, device, scale, bg = args

                if self.timing:
                    self.start_recorder.record()
                x = self.texture(uv_map[:,0,...])
                x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                assert x.shape[1] >= 12
                basis = self._spherical_harmonics_basis(extrinsics[:,0,...])
                basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
                x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis
                if self.timing: 
                    self.end_recorder.record()  
                    torch.cuda.synchronize()
                    t1_gridsample = self.start_recorder.elapsed_time(self.end_recorder)
                            
            if self.timing:
                self.start_recorder.record()
            y, cache, skip_connection = self.unet(x) # BG is NOT used if self.predict_mask == True
            if self.timing: 
                    self.end_recorder.record()  
                    torch.cuda.synchronize()
                    t1_generator = self.start_recorder.elapsed_time(self.end_recorder)

            if self.timing:
                self.start_recorder.record()
            if self.use_pose:
                if self.sh_pose:
                    skip_basis_2 = self._spherical_harmonics_basis(pose[:,1,:3])
                    if self.use_exp:
                        mlp_input =  torch.cat((skip_basis_2, pose[:,1,...], (pose[:,1,...]-pose[:,0,...]) * 1., (extrinsics[:,1,:2]-extrinsics[:,0,:2])*1., extrinsics[:,1,:2], exp[:,1,...], (exp[:,1,...]-exp[:,0,...])*1. ), dim=-1)
                    else:
                        mlp_input =  torch.cat((skip_basis_2, pose[:,1,...], (pose[:,1,...]-pose[:,0,...]) * 1., (extrinsics[:,1,:2]-extrinsics[:,0,:2])*1., extrinsics[:,1,:2]), dim=-1)
                else:
                    mlp_input =  torch.cat(((pose[:,1,...]-pose[:,0,...]) * 1., (extrinsics[:,1,:2]-extrinsics[:,0,:2])*1., extrinsics[:,1,:2]), dim=-1)

                if self.use_mlp:
                    mlp_out = self.mlp(mlp_input)
                    mlp_out = mlp_out[..., None, None].repeat(1,1,cache.shape[-2], cache.shape[-1])
                else:
                    mlp_out = mlp_input
                    mlp_out = mlp_out[..., None, None].repeat(1,1,cache.shape[-2], cache.shape[-1])
                    mlp_forward_pass = 0

                cache = torch.cat((cache, mlp_out), dim=1)

            if self.timing: 
                self.end_recorder.record()  
                torch.cuda.synchronize()
                mlp_forward_pass = self.start_recorder.elapsed_time(self.end_recorder)

            if self.timing:
                self.start_recorder.record()
            if not self.use_pose:
                pose = None
            y2, secondary_output = self.warpnet(cache, uv_map[:,1,...], extrinsics[:,1,...], scale, bg[:,1,...], uv_map[:,0,...], skip_connection, self.texture, device, y, pose)
            if self.timing: 
                    self.end_recorder.record()  
                    torch.cuda.synchronize()
                    t2_warp = self.start_recorder.elapsed_time(self.end_recorder)

            if not self.predict_mask:
                return x[:, 0:3, :, :], y, y2, secondary_output
            else:
                pred_rgb_1 = y[:, 0:3, ...]
                pred_mask_1 = torch.sigmoid(y[:, 3, ...]).unsqueeze(1)
                pred_rgb_2 = y2[:, 0:3, ...]
                pred_mask_2 = torch.sigmoid(y2[:, 3, ...]).unsqueeze(1)
            
                if self.timing:
                    return x[:, 0:3, :, :], pred_rgb_1, pred_mask_1, pred_rgb_2, pred_mask_2, secondary_output, [t1_gridsample, t1_generator, mlp_forward_pass, t2_warp]
                else:
                    return x[:, 0:3, :, :], pred_rgb_1, pred_mask_1, pred_rgb_2, pred_mask_2, secondary_output

            