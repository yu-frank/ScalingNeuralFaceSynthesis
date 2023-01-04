import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

from util import augment
import h5py
import torchvision.transforms as transforms
from skimage.transform import estimate_transform, warp

class WarpUVDataset(Dataset):

    def __init__(self, dir, H, W, split, view_direction=True, use_aug=True, 
                use_mask=True, fix_background=False, t_diff=2, 
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], use_pose=False, normSplit=None,
                randomFlip=False, use_exp=True, float_image=True):

        """
        dir (string): path to the dataset (.h5 file)
        H (int): Crop image height (0 is no cropping)
        W (int): Crop iamge width (0 is no cropping)
        split (string): which split of the dataset to load {train, validation, test}
        view_direction (bool): load DECA camera extrinsic parameters from the dataset (default: True)
        use_aug (bool): perform image augmentation during training (default: True)
        use_mask (bool): load image foreground masks during training (for the Neural Texture Loss) (default: True)
        fix_background (bool): whether or not to return a static background (for novel view synthesis) (default: False)
        t_diff (int): indicates how many frames to leave between warping data pairs (default: 2)
        mean (list[float]): img color mean for the dataset (default: ImageNet)
        std (list[float]): img color standard deviation for the dataset (default: ImageNet)
        use_pose (bool): return the pose information from the dataset
        normSplit (string): select which dataset partition is used for normalizing the dataset, None means
        to use the same dataset as the one being loaded (default: None)
        randomFlip (bool): whether or not to swap t=0 and t=t+t_diff in training (default: False)
        use_exp (bool): load the FLAME expression parameters from the dataset (default: True)
        float_image (bool): are the images in the dataset [0,1] (float) or [0,255] (int) (default: True)
        """
        
        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction
        self.use_aug = use_aug
        self.use_mask = use_mask
        self.split = split 
        self.fix_background = fix_background
        self.t_diff = t_diff
        self.use_pose = use_pose
        self.float_image = float_image

        # default values are from ImageNet
        self.mean = mean
        self.std = std

        self.randomFlip = randomFlip

        # use the expressions from DECA
        self.use_exp = use_exp

        self.transform = transforms.ToPILImage()

        self.median_background = os.path.join(dir, 'median_background.jpg')

        if self.use_pose:
            self.pose_dir = os.path.join(dir, "{}-pose.h5".format(split))
            self.pose_len = h5py.File(self.pose_dir,'r')['pose'].shape[0]

        if self.use_exp:
            self.exp_dir = os.path.join(dir, "{}-exp.h5".format(split))
            self.exp_len = h5py.File(self.exp_dir, 'r')['exp'].shape[0]

        self.data_dir = os.path.join(dir,"{}.h5".format(split))

        self.data_len = h5py.File(self.data_dir,'r')['bbox_center'].shape[0]
        if self.use_mask:
            self.mask_dir = os.path.join(dir,"{}-masks.h5".format(split))
            self.mask_len = h5py.File(self.mask_dir,'r')['mask'].shape[0]
            assert self.data_len == self.mask_len
        
        if self.use_pose:
            assert self.data_len == self.pose_len
        
        if self.use_exp:
            assert self.data_len == self.exp_len

        self.normSplit = normSplit
        # Compute the Pose and Extrinsic Mean and STD 
        if self.use_pose:
            if self.normSplit == None:
                all_poses = h5py.File(self.pose_dir,'r')['pose']
            else:
                self.normPose_dir = os.path.join(dir, "{}-pose.h5".format(normSplit))
                all_poses = h5py.File(self.normPose_dir,'r')['pose']
            self.pose_mean = np.mean(all_poses, axis=0)[0]
            self.pose_std = np.std(all_poses, axis=0)[0]
        
        if self.view_direction:
            if self.normSplit == None:

                all_extrinsics = h5py.File(self.data_dir,'r')['extrinsics']
            else:
                self.normEx_dir = os.path.join(dir, "{}.h5".format(normSplit))
                all_extrinsics = h5py.File(self.normEx_dir,'r')['extrinsics']
            self.ex_mean = np.mean(all_extrinsics, axis=0)
            self.ex_std = np.std(all_extrinsics, axis=0)

        if self.use_exp:
            if self.normSplit == None:
                all_exp = h5py.File(self.exp_dir, 'r')['exp']
            else:
                self.normExp_dir = os.path.join(dir, "{}-exp.h5".format(normSplit))
                all_exp = h5py.File(self.normExp_dir, 'r')['exp']
            self.expression_mean = np.mean(all_exp, axis=0)[0]
            self.expression_std = np.std(all_exp, axis=0)[0]

    def __len__(self):
        return self.data_len - self.t_diff

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
        bg = np.asarray(bg)
        if not self.fix_background:
            size = h5py.File(self.data_dir,'r')['bbox_size'][idx]
            center = h5py.File(self.data_dir,'r')['bbox_center'][idx].astype(int)
        else:
            # default to set the background to the cropped background of the first frame in the dataset
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
            # for image t+t_diff (crop this image the same as the previous (t) image)
            img, uv_map, img_mask, bg = augment(img, uv_map, self.crop_size, self.use_aug, img_mask, bg=bg, given_crop=given_crop, mean=self.mean, std=self.std)
        elif return_crop:
            # for image t (return the random crop so the image at t+t_diff can be cropped the same)
            img, uv_map, img_mask, bg, w1, h1, crop_w, crop_h = augment(img, uv_map, self.crop_size, self.use_aug, img_mask, bg=bg, return_aug_crop=True, mean=self.mean, std=self.std)
            aug_crop_parameters = [w1, h1, crop_w, crop_h]
        else:
            # used if there is no augmentation
            img, uv_map, img_mask, bg = augment(img, uv_map, self.crop_size, self.use_aug, img_mask, bg=bg, mean=self.mean, std=self.std)

        # clip uv_map
        uv_map = torch.clip(uv_map, -1, 1)

        if self.view_direction:
            extrinsics = h5py.File(self.data_dir,'r')['extrinsics'][idx]
            extrinsics[:2] = (extrinsics[:2] - self.ex_mean[:2])/self.ex_std[:2]
            extrinsics[2] = 1

            if return_crop:
                return_dict = {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg, 'extrinsics': extrinsics, 'crop_center': center, 'crop_size': size, 'aug_crop_parameters': aug_crop_parameters}
            else:
                return_dict = {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg, 'extrinsics': extrinsics, 'crop_center': center, 'crop_size': size}
        else:
            if return_crop:
                return_dict = {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg, 'crop_center': center, 'crop_size': size, 'aug_crop_parameters': aug_crop_parameters}
            else:
                return_dict = {'img': img, 'uv_map': uv_map, 'img_mask': img_mask, 'bg': bg}

        if self.use_pose:
            # normalize the extrinsics
            return_dict['pose'] = (h5py.File(self.pose_dir,'r')['pose'][idx] - self.pose_mean)/self.pose_std

        if self.use_exp:
            return_dict['exp'] = (h5py.File(self.exp_dir, 'r')['exp'][idx] - self.expression_mean)/self.expression_std

        return return_dict

    def __getitem__(self, idx) -> dict:
        """
        0 index is the first example, 
        1 index is the second example
        """
        # Get Data from 2 examples
        if self.use_aug: # need to store the crop from the first one and then use it in the second!
            if np.random.randint(0,2) and self.randomFlip:
                img1_dict = self.__getOne__(idx, return_crop=True) # the "temporal" input
                aug_crop_parameters = img1_dict['aug_crop_parameters'] # should be a list [w1, h1, crop_w, crop_h]
                img2_dict = self.__getOne__(idx+np.random.randint(1, self.t_diff + 1), given_crop=aug_crop_parameters) # the one we want to predict
            else:
                img1_dict = self.__getOne__(idx+np.random.randint(1, self.t_diff + 1), return_crop=True) # the "temporal" input
                aug_crop_parameters = img1_dict['aug_crop_parameters'] # should be a list [w1, h1, crop_w, crop_h]
                img2_dict = self.__getOne__(idx, given_crop=aug_crop_parameters) # the one we want to predict
        else:
            if np.random.randint(0,2) and self.randomFlip: 
                img1_dict = self.__getOne__(idx)
                img2_dict = self.__getOne__(idx+np.random.randint(1, self.t_diff + 1))
            else:
                img1_dict = self.__getOne__(idx)
                img2_dict = self.__getOne__(idx+np.random.randint(1, self.t_diff + 1))

        # concatenate these dictionaries
        result_dict = {
            'img': torch.cat((img1_dict['img'].unsqueeze(0), img2_dict['img'].unsqueeze(0))),
            'uv_map': torch.cat((img1_dict['uv_map'].unsqueeze(0), img2_dict['uv_map'].unsqueeze(0))),
            'img_mask': torch.cat((img1_dict['img_mask'].unsqueeze(0), img2_dict['img_mask'].unsqueeze(0))),
            'bg': torch.cat((img1_dict['bg'].unsqueeze(0), img2_dict['bg'].unsqueeze(0))),
        }
        if self.view_direction:
            result_dict['extrinsics'] = torch.cat((torch.FloatTensor(img1_dict['extrinsics']).unsqueeze(0), torch.FloatTensor(img2_dict['extrinsics']).unsqueeze(0)))

        if self.use_pose:
            result_dict['pose'] = torch.cat((torch.FloatTensor(img1_dict['pose']), torch.FloatTensor(img2_dict['pose'])))

        if self.use_exp:
            result_dict['exp'] = torch.cat((torch.FloatTensor(img1_dict['exp']), torch.FloatTensor(img2_dict['exp'])))

        return result_dict
