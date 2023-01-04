import numpy as np
import os
import torch
from model.inference_pipeline import InferencePipeLine
import util
import losses
import json
import h5py

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_mean_std(args):
    stats = util.DATSET_STATS[args.dataset_name]
    stats = stats.split(',')
    stats = [float(x) for x in stats]
    mean = stats[:3]
    std = stats[3:]
    return mean, std

class InferenceRunner():
    def __init__(self, model_path, mean=None, std=None, checkpoint_name='best_val_psnr.tar', device='cuda', timing=True) -> None:
        """
        Inputs: 
            model_path (String): path to the personalized model folder [Folder Structure: "model_path"/checkpoints/"checkpoint_name"]
            checkpoint_name (String): which checkpoint in the model folder should be used for inference (default: 'best_val_psnr.tar')
            device (String): whether to run model on CPU or GPU (default: 'cuda')
            timing (Boolean): whether or not to time the components of the network
            V2_cache (Int): how many warpings to do before caching again
        """
        # Specify whether or not we are timing
        self.timing = False
        self.device = device

        # load model config
        self.model_folder = model_path
        with open(os.path.join(self.model_folder, 'commandline_args.txt')) as f:
            args = json.load(f)
        args = dotdict(args)

        # Specify Mean and STD for denormalizing images
        if mean is None or std is None:
            self.mean, self.std = get_mean_std(args)
        else:
            self.mean, self.std = mean, std

        self.mean = torch.FloatTensor(self.mean).to(self.device)
        self.std = torch.FloatTensor(self.std).to(self.device)

        # Specify whether or not we use the pose
        self.use_pose = args.use_pose
        if args.use_exp is None:
            self.use_exp = 0 # should be 'frank'
        else:
            self.use_exp = args.use_exp

        # Get the Model Checkpoint
        checkpoint = torch.load(os.path.join(model_path, 'checkpoints', checkpoint_name), map_location='cpu')

        if self.timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        else:
            start, end = None, None

        
        self.predict_mask = args.predict_mask
        self.decoder_image_size = 512

        print('network size: ', args.network_size)
        print('t diff: ', args.t_diff)
        
        model = InferencePipeLine(args.texturew, args.textureh, args.texture_dim, \
                        args.use_pyramid, args.view_direction, args.use_lowpass, \
                        args.use_upconv, args.low_pass_sigma, device, use_bg=args.use_bg, \
                        use_instance_norm=args.use_instance_norm, use_warp=args.use_warp, \
                        network_size=args.network_size, predict_mask=args.predict_mask, cache_layer=args.cache_layer, concat_uv_diff=args.concat_uv_diff,
                        use_pose=args.use_pose, warp_sample=args.warp_sample, cache_features=args.cache_features, 
                        sh_skips=args.sh_skips, sh_pose=args.sh_pose, use_mlp=args.use_mlp, skip_cache_dim=args.skip_cache_dim, timing=timing, 
                        start_recorder=start, end_recorder=end, use_exp=self.use_exp)  

        test = torch.nn.DataParallel(model) # HACK
        test.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = test.module
        self.model = self.model.to(device)
        self.model.eval()

        self.L1 = torch.nn.L1Loss()
        print('done')

    def list_2_codedict(self, l: list) -> dict:
        return {
            'shape': torch.FloatTensor(l[:100]).unsqueeze(0),
            'exp': torch.FloatTensor(l[100:150]).unsqueeze(0),
            'pose': torch.FloatTensor(l[150:156]).unsqueeze(0),
            'cam': torch.FloatTensor(l[156:159]).unsqueeze(0),
            'light': torch.FloatTensor(l[159:159+27]).reshape(9,3).unsqueeze(0),
            'detail': torch.FloatTensor(l[159+27:314]).unsqueeze(0),
        }, l[-3:]
    
    def move2GPU(self, d: dict):
        for key in d.keys():
            d[key] = d[key].to(self.device)
        return d

    def to_device(self, input) -> dict:
        """
        Putting all data in input dictionary onto the proper device
        """
        for key, value in input.items():
            input[key] = value.to(self.device)

        return input

    def calculate_losses(self, prediction, ground_truth) -> dict:
        batch_l1_1 = self.L1(torch.clip(util.denormalize_img(prediction.cpu().permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1), torch.clip(util.denormalize_img(ground_truth.permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1))
        batch_psnr_1 = losses.PSNR(torch.clip(util.denormalize_img(prediction.cpu().permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1), torch.clip(util.denormalize_img(ground_truth.permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1))
        batch_ssim_1 = losses.SSIM(torch.clip(util.denormalize_img(prediction.cpu().permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1), torch.clip(util.denormalize_img(ground_truth.permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1))
        return {'l1': batch_l1_1,
                'psnr': batch_psnr_1,
                'ssim': batch_ssim_1
        }
    
    def denormalize_img(self, img):
        clip_preds = (torch.clip(util.denormalize_img(img.cpu().permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1).permute(0,3,1,2)).to(self.device)
        clip_preds = (clip_preds * 255).cpu().permute(0,2,3,1)
        return clip_preds.detach().squeeze().cpu().numpy().astype('uint8')

    def generate_metric_img(self, img, mask, background):
        metric_img = mask * img + (1. - mask) * background.to(self.device)
        metric_img = metric_img.detach().cpu()
        return metric_img

    def generate_rgb_img_cuda(self, img, mask, background):
        background = background.repeat(img.shape[0], 1, 1, 1)
        clip_preds1 = torch.clip(util.denormalize_img_cuda(img.permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1).permute(0,3,1,2)
        mask = torch.round(mask)
        clip_preds1 = mask * clip_preds1 + (1. - mask) * background
        clip_preds1 = (clip_preds1 * 255).permute(0,2,3,1)
        return clip_preds1.cpu().numpy().astype('uint8')

    def generate_background_img(self, img, mask, background):
        """
        Returns the image used for calculating validation metrics 
        """
        m_img = mask * img + (1. - mask) * background.to(self.device)
        m_img = (torch.clip(util.denormalize_img(m_img.cpu().permute(0,2,3,1), mean=self.mean, std=self.std),min=0, max=1).permute(0,3,1,2)).detach().cpu().squeeze().permute(1,2,0).numpy()
        m_img *= 255
        m_img = m_img.astype('uint8')
        return m_img

    def generator(self, input):
        """
        Updates the Cache of the model
        INPUTS
        input (dictionary): input parameters
        """
        self.model.update_cache(input)

    def warp(self, input):
        """
        Runs the forward pass for the model given that a cache has already been used
        INPUTS
        input (dictionary): input parameters
        """
        if self.predict_mask:
            img, mask = self.model(input)
            return img, mask
        else:
            img = self.model(input)
            return img
        
        
