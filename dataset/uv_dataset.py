import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

from util import augment
import h5py
import torchvision.transforms as transforms
from skimage.transform import estimate_transform, warp


class UVDataset(Dataset):

    def __init__(self, dir, H, W, split, view_direction=False, 
                use_aug=True, use_mask=True, 
                fix_background=False, single_temporal=True,
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], float_image=True):

        """
        dir (string): path to the dataset (.h5 file)
        H (int): Crop image height (0 is no cropping)
        W (int): Crop iamge width (0 is no cropping)
        split (string): which split of the dataset to load {train, validation, test}
        view_direction (bool): load DECA camera extrinsic parameters from the dataset (default: True)
        use_aug (bool): perform image augmentation during training (default: True)
        use_mask (bool): load image foreground masks during training (for the Neural Texture Loss) (default: True)
        fix_background (bool): whether or not to return a static background (for novel view synthesis) (default: False)
        single_temporal (bool): load information from the previous frame (t-1) (default: True)
        mean (list[float]): img color mean for the dataset (default: ImageNet)
        std (list[float]): img color standard deviation for the dataset (default: ImageNet)
        float_image (bool): are the images in the dataset [0,1] (float) or [0,255] (int) (default: True)
        """
        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction
        self.use_aug = use_aug
        self.use_mask = use_mask
        self.split = split 
        self.fix_background = fix_background
        self.single_temporal = single_temporal
        self.float_image = float_image

        self.mean = mean 
        self.std = std 

        self.transform = transforms.ToPILImage()

        self.median_background = os.path.join(dir, 'median_background.jpg')

        self.data_dir = os.path.join(dir,"{}.h5".format(split))
        self.data_len = h5py.File(self.data_dir,'r')['bbox_center'].shape[0]
        if self.use_mask:
            self.mask_dir = os.path.join(dir,"{}-masks.h5".format(split))
            self.mask_len = h5py.File(self.mask_dir,'r')['mask'].shape[0]
            assert self.data_len == self.mask_len


    def __len__(self):
        if self.single_temporal:
            return self.data_len - 1
        else:
            return self.data_len

    def __getOne__(self, idx, return_crop=False, given_crop=None):
        if self.float_image:
            img = self.transform(torch.FloatTensor(h5py.File(self.data_dir,'r')['larger_image'][idx]))
        else:
            img = self.transform(torch.FloatTensor(h5py.File(self.data_dir,'r')['larger_image'][idx]) / 255.)
        uv_map = h5py.File(self.data_dir,'r')['grid'][idx]

        if self.use_mask:
            img_mask = np.expand_dims(h5py.File(self.mask_dir,'r')['mask'][idx], axis=-1)
        else:
            img_mask = np.ones((uv_map.shape[0], uv_map.shape[1], 1), dtype=float)
        
        bg = Image.open(self.median_background, 'r')
            
        if bg is not None:
            bg = np.asarray(bg)
            if not self.fix_background:
                size = h5py.File(self.data_dir,'r')['bbox_size'][idx]
                center = h5py.File(self.data_dir,'r')['bbox_center'][idx].astype(int)
            else:
                size = h5py.File(self.data_dir,'r')['bbox_size'][0]
                center = h5py.File(self.data_dir,'r')['bbox_center'][0].astype(int)

            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS_LARGE = np.array([[0,0], [0,img.size[0] - 1], [img.size[1] - 1, 0]])
            tform_large = estimate_transform('similarity', src_pts, DST_PTS_LARGE)
            bg = warp(bg, tform_large.inverse, output_shape=(img.size[0],img.size[1]))    
        
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')

        if given_crop is not None:
            # not finished
            img, uv_map, img_mask, bg = augment(img, uv_map, self.crop_size, self.use_aug, img_mask, bg=bg, given_crop=given_crop, mean=self.mean, std=self.std)
        elif return_crop:
            # not finished
            img, uv_map, img_mask, bg, w1, h1, crop_w, crop_h = augment(img, uv_map, self.crop_size, self.use_aug, img_mask, bg=bg, return_aug_crop=True, mean=self.mean, std=self.std)
            aug_crop_parameters = [w1, h1, crop_w, crop_h]
        else:
            # finished
            img, uv_map, img_mask, bg = augment(img, uv_map, self.crop_size, self.use_aug, img_mask, bg=bg, mean=self.mean, std=self.std)

        # clip uv_map
        uv_map = torch.clip(uv_map, -1, 1)

        if self.view_direction:
            extrinsics = h5py.File(self.data_dir,'r')['extrinsics'][idx]
            if return_crop:
                return {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg, 'extrinsics': extrinsics, 'crop_center': center, 'crop_size': size, 'aug_crop_parameters': aug_crop_parameters}
            else:
                return {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg, 'extrinsics': extrinsics, 'crop_center': center, 'crop_size': size}
        else:
            if return_crop:
                return {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg, 'crop_center': center, 'crop_size': size, 'aug_crop_parameters': aug_crop_parameters}
            else:
                return {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg}

    def __getitem__(self, idx) -> dict:
        if not self.single_temporal:
            return self.__getOne__(idx)
        else:
            """
            0 index is the first example, 
            1 index is the second example # the one we're trying to predict!
            """
            if self.use_aug: # need to store the crop from the first one and then use it in the second!
                img1_dict = self.__getOne__(idx, return_crop=True) # the "temporal" input
                aug_crop_parameters = img1_dict['aug_crop_parameters'] # should be a list [w1, h1, crop_w, crop_h]
                img2_dict = self.__getOne__(idx+1, given_crop=aug_crop_parameters) # the one we want to predict
            else:
                img1_dict = self.__getOne__(idx) # the "temporal" input
                img2_dict = self.__getOne__(idx+1) # the one we want to predict

            # concatenate these dictionaries
            result_dict = {
                'img': img2_dict['img'],
                'uv_map': torch.cat((img1_dict['uv_map'].unsqueeze(0), img2_dict['uv_map'].unsqueeze(0))),
                'img_mask': img2_dict['img_mask'],
                'bg':img2_dict['bg'],
                'crop_center':img2_dict['crop_center'],
                'crop_size':img2_dict['crop_size']
            }
            if self.view_direction:
                result_dict['extrinsics'] = torch.cat((torch.FloatTensor(img1_dict['extrinsics']).unsqueeze(0), torch.FloatTensor(img2_dict['extrinsics']).unsqueeze(0)))

            return result_dict