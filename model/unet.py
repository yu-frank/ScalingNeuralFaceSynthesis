import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class down(nn.Module):
    def __init__(self, in_ch, out_ch, position=None):
        super(down, self).__init__()
        if position == 'outermost':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),  # HALVES THE DIMENSION (HxW)
            )
        elif position == 'innermost':
            self.conv = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),  # HALVES THE DIMENSION (HxW)                
            )
        else:
            self.conv = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),  # HALVES THE DIMENSION (HxW)
                nn.InstanceNorm2d(out_ch)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, output_pad=0, concat=True, position=None):
        super(up, self).__init__()
        self.concat = concat
        if position == 'outermost':
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.Tanh()
            )
        elif position == 'innermost':
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.InstanceNorm2d(out_ch),
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.InstanceNorm2d(out_ch),
            )
    def forward(self, x1, x2):
        if self.concat:
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv(x1)
        return x1

class upconv(nn.Module):
    def __init__(self, in_ch, out_ch, output_pad=0, concat=True, position=None):
        super(upconv, self).__init__()
        self.concat = concat
        if position == 'outermost':
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            )
        elif position == 'innermost':
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_ch),
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_ch),
            )
    def forward(self, x1, x2):
        if self.concat:
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv(x1)
        return x1

class lowpass(nn.Module):
    def __init__(self, kernel_size, padding, sigma, device):
        super(lowpass, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = padding

        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        self.low_pass_filter = gaussian_kernel.view(kernel_size, kernel_size)
    
    def forward(self, x):
        x = F.conv2d(x, self.low_pass_filter.expand(x.size(1), 1, self.kernel_size, self.kernel_size).to(x.device), stride=1, groups=x.size(1), padding=self.padding)
        return x

class UNet_Mask(nn.Module):
    def __init__(self, input_channels, output_channels, use_lowpass=None, device=None, \
                use_upconv=False, sigma=2, use_bg=None, use_warp=False, use_instance_norm=False, \
                network_size=1, cache_layer=4, cache_features=False, skip_cache_dim=9,
                use_cache_layers=[3,4,5]):
        super(UNet_Mask, self).__init__()

        assert output_channels == 4
        self.network_size = network_size
        self.output_channels = output_channels
        
        self.use_cache_layers = use_cache_layers

        self.down1 = down(input_channels, 64 * self.network_size, position='outermost')
        self.down2 = down(64 * self.network_size, 128 * self.network_size)
        self.down3 = down(128 * self.network_size, 256 * self.network_size)
        self.down4 = down(256 * self.network_size, 512 * self.network_size)
        self.down5 = down(512 * self.network_size, 512 * self.network_size, position='innermost')

        self.use_bg = use_bg
        self.use_warp = use_warp
        self.use_instance_norm = use_instance_norm
        
        self.cache_layer = cache_layer

        self.cache_features = cache_features
        
        assert self.cache_layer == 3 or self.cache_layer == 4

        if self.cache_layer == 4:
            self.cache_dim = 128 * self.network_size
        elif self.cache_layer == 3:
            self.cache_dim = 256 * self.network_size

        
        self.cache_channel_size = skip_cache_dim

        if not use_upconv:
            # use transpose conv
            self.up1 = up(512 * self.network_size, 512 * self.network_size, concat=False, position='innermost')
            self.up2 = up(1024 * self.network_size, 512 * self.network_size)
            self.up3 = up(768 * self.network_size, 256 * self.network_size)
            self.up4 = up(384 * self.network_size, 128 * self.network_size)
            self.up5 = up(192 * self.network_size, output_channels + self.cache_channel_size * self.cache_features*(self.use_warp), position='outermost')

        else:
            self.up1 = upconv(512 * self.network_size, 512 * self.network_size, concat=False, position='innermost')
            self.up2 = upconv(1024 * self.network_size, 512 * self.network_size)
            self.up3 = upconv(768 * self.network_size, 256 * self.network_size)
            self.up4 = upconv(384 * self.network_size, 128 * self.network_size)
            self.up5 = upconv(192 * self.network_size, output_channels + self.cache_channel_size * self.cache_features*(self.use_warp), position='outermost')

        self.device = device
        self.sigma = sigma
        self.useLowpass = use_lowpass
        if self.useLowpass:
            self.lowpass = lowpass(kernel_size=3, padding=1, sigma=self.sigma, device=device)

    def forward(self, x):
        skip_connection = []
        x1 = self.down1(x)

        x2 = self.down2(x1)

        x3 = self.down3(x2)

        x4 = self.down4(x3)

        x5 = self.down5(x4)


        if self.useLowpass:
            x5 = self.lowpass(x5)

        x = self.up1(x5, None)

        x = self.up2(x, x4)

        x = self.up3(x, x3)

        if self.use_warp and self.cache_layer == 3:
            if 3 in self.use_cache_layers:
                cache = x.clone()
            else:
                cache = torch.zeros_like(x).to(x.device)
            if not self.cache_features:
                skip_connection.append(x2.clone()) 
                skip_connection.append(x1.clone())

        x = self.up4(x, x2)
        if self.cache_features:
            # cache the remainder
            features_4 = F.interpolate(x[:,-self.cache_channel_size:,...].clone(), scale_factor=0.5, mode='bilinear', align_corners=False)
            if 4 in self.use_cache_layers:
                skip_connection.append(features_4)
            else:
                skip_connection.append(torch.zeros_like(features_4).to(features_4.device))

        if self.use_warp and self.cache_layer == 4:
            cache = x.clone()
            if not self.cache_features:
                skip_connection.append(x1.clone())

        x_temp5 = self.up5(x, x1)
        if self.cache_features:
            x = x_temp5[:,:self.output_channels, ...].clone()
            features_5 = F.interpolate(x_temp5[:,self.output_channels:,...].clone(), scale_factor=0.5, mode='bilinear', align_corners=False)
            if 5 in self.use_cache_layers:
                skip_connection.append(features_5)
            else:
                skip_connection.append(torch.zeros_like(features_5).to(features_5.device))
        else:
            x = x_temp5
        if self.use_warp:
            return x, cache, skip_connection
        else:
            return x


        
    
