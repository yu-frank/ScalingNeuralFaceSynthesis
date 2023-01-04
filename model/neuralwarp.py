import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.unet import upconv
import numpy as np

class warp_conv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(warp_conv, self).__init__()
        self.layer = torch.nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            
        )
    
    def forward(self, x):
        return self.layer(x)

class NeuralWarp_Mask(nn.Module):
    def __init__(self, input_dims, output_dims, cache_layer=4,
                concat_uv_diff=False, use_pose=False, use_pose_mlp_dim=32,
                warp_sample=True, cache_features=False, sh_skips=False, skip_cache_dim=9):
        super(NeuralWarp_Mask, self).__init__()
        self.cache_layer = cache_layer
        self.concat_uv_diff = concat_uv_diff
        self.warp_uv = 1
        self.warp_sample = warp_sample
        self.cache_features = cache_features
        self.sh_skips = sh_skips

        if self.cache_features:
                self.skip_connection_size_0 = skip_cache_dim
                self.skip_connection_size_1 = skip_cache_dim
        else:
            self.skip_connection_size_0 = input_dims//2
            self.skip_connection_size_1 = input_dims//4

        if self.cache_layer == 4:
            if self.concat_uv_diff:
                self.conv1 = upconv(input_dims + self.skip_connection_size_0 + 16 * self.warp_uv * self.warp_sample + 2 * self.warp_uv * self.concat_uv_diff + use_pose_mlp_dim * use_pose, output_dims, position='outermost') # add the uv dimensions (2)
            else:
                self.conv1 = upconv(input_dims + self.skip_connection_size_0 + 16 * self.warp_uv * self.warp_sample + use_pose_mlp_dim * use_pose, output_dims, position='outermost')
        else:
            if self.concat_uv_diff:
                self.conv1 = upconv(input_dims + self.skip_connection_size_0 + 16 * self.warp_uv * self.warp_sample + 2 * self.warp_uv * self.concat_uv_diff + use_pose_mlp_dim * use_pose, input_dims//2, position=None)
            else:
                self.conv1 = upconv(input_dims + self.skip_connection_size_0 + 16 * self.warp_uv * self.warp_sample + use_pose_mlp_dim * use_pose, input_dims//2, position=None)   # sampled features  

            self.conv2 = upconv(input_dims//2 + self.skip_connection_size_1, output_dims, position="outermost")
        
        self.align_corners = False 
        self.padding_mode = 'border'
        

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

    def forward(self, nn_features, uv_map_2, extrinsics_2, scale, bg, uv_map_1, skip_connection, neural_texture, device, predict_t1, pose):
        cache_sample = nn_features

        if self.sh_skips and self.cache_features:
            skip_basis_1 = self._spherical_harmonics_basis(pose[:,0,:3])
            skip_basis_1 = skip_basis_1.view(skip_basis_1.shape[0], skip_basis_1.shape[1], 1, 1)
            skip_basis_2 = self._spherical_harmonics_basis(pose[:,1,:3])
            skip_basis_2 = skip_basis_2.view(skip_basis_2.shape[0], skip_basis_2.shape[1], 1, 1)

            for s in range(len(skip_connection)):
                if self.skip_connection_size_1 == 9: #to support old stuff...
                    skip_connection[s] = skip_connection[s] * (skip_basis_2 - skip_basis_1)
                elif self.skip_connection_size_1 == 8:
                    skip_connection[s] = skip_connection[s] * (skip_basis_2 - skip_basis_1)[:,1:,...]
                else:
                    skip_connection[s][:,:9,...] = skip_connection[s][:,:9,...] * (skip_basis_2 - skip_basis_1)
        
        if self.cache_layer == 4:
            if self.warp_sample:
                x = neural_texture(uv_map_2)
                x = F.interpolate(x, scale_factor=nn_features.shape[-1]/x.shape[-1], mode='bilinear', align_corners=False)
                assert x.shape[1] >= 12
                basis = self._spherical_harmonics_basis(extrinsics_2)
                basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
                x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis

                cache_sample = torch.cat((cache_sample, x), dim=1)

            if self.concat_uv_diff:
                uv_diff = (uv_map_2 - uv_map_1).permute(0,3,1,2)
                uv_diff = F.interpolate(uv_diff, scale_factor=nn_features.shape[-1]/uv_diff.shape[-1], mode='bilinear', align_corners=False)
                cache_sample = torch.cat((cache_sample, uv_diff), dim=1)

            x1 = self.conv1(cache_sample, skip_connection[0])
            return x1, cache_sample

        # if layer = 3
        else:
            if self.warp_sample:
                x = neural_texture(uv_map_2)

                x = F.interpolate(x, scale_factor=nn_features.shape[-1]/x.shape[-1], mode='bilinear', align_corners=False)
                assert x.shape[1] >= 12
                basis = self._spherical_harmonics_basis(extrinsics_2)
                basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
                x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis.to(x.device)
            
                cache_sample = torch.cat((cache_sample, x), dim=1)
            if self.concat_uv_diff:
                uv_diff = (uv_map_2 - uv_map_1).permute(0,3,1,2)
                uv_diff = F.interpolate(uv_diff, scale_factor=nn_features.shape[-1]/uv_diff.shape[-1], mode='bilinear', align_corners=False)
                cache_sample = torch.cat((cache_sample, uv_diff), dim=1)
            
            x1 = self.conv1(cache_sample, skip_connection[0]) # layer 4
            x1 = self.conv2(x1, skip_connection[1]) # layer 5
            return x1, cache_sample