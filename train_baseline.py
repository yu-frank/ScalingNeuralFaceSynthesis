import argparse
import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.uv_dataset import UVDataset
from dataset.uv_dataset import UVDataset
import util
from train_logging import Logger
import config
from model.pipeline import PipeLine
from validation import validation_baseline
from model.vgg import VGGPerceptualLoss

parser = argparse.ArgumentParser()
parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save checkpoint')
parser.add_argument('--epoch', type=int, default=config.EPOCH)
parser.add_argument('--cropw', type=int, default=config.CROP_W)
parser.add_argument('--croph', type=int, default=config.CROP_H)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
parser.add_argument('--betas', type=str, default=config.BETAS)
parser.add_argument('--l2', type=str, default=config.L2_WEIGHT_DECAY)
parser.add_argument('--eps', type=float, default=config.EPS)
parser.add_argument('--load', type=str, default=config.LOAD)
parser.add_argument('--load_step', type=int, default=config.LOAD_STEP)
parser.add_argument('--epoch_per_checkpoint', type=int, default=config.EPOCH_PER_CHECKPOINT)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--low_pass_sigma', type=float, default=config.LOWPASS_SIGMA)
parser.add_argument('--unet_decay', type=float, default=config.UNET_DECAY)
parser.add_argument('--use_lowpass', type=int, default=config.USE_LOWPASS)
parser.add_argument('--use_mask', type=int, default=config.USE_MASK)
parser.add_argument('--use_upconv', type=int, default=config.USE_UPCONV)
parser.add_argument('--use_aug', dest='useAug', type=int, default=config.USE_AUG)
parser.add_argument('--use_warp', type=int, default=config.USE_WARP)
parser.add_argument('--use_instance_norm', type=int, default=config.USE_INSTANCE_NORM)
parser.add_argument('--view_direction', type=int, default=config.VIEW_DIRECTION)
parser.add_argument('--our_model', type=int, default=config.OUR_MODEL)
parser.add_argument('--use_lr_decay', type=int, default=config.USE_LR_DECAY)
parser.add_argument('--train', type=str, default=config.TRAIN)
parser.add_argument('--validation', type=str, default=config.VALIDATION)
parser.add_argument('--test', type=str, default=config.TEST)
parser.add_argument('--network_size', type=int, default=config.NETWORK_SIZE)
parser.add_argument('--predict_mask', type=int, default=config.PREDICT_MASK)
parser.add_argument('--use_tensorboard', type=int, default=config.USE_TENSORBOARD)
parser.add_argument('--cache_layer', type=int, default=config.CACHE_LAYER)
parser.add_argument('--t_diff', type=int, default=config.T_DIFF)
parser.add_argument('--dataset_name', type=str, default=config.DATASET_NAME)
parser.add_argument('--use_pose', type=int, default=config.USE_POSE)
parser.add_argument('--nt_coef', type=float, default=config.NT_COEF)
parser.add_argument('--random_flip', type=int, default=config.RANDOM_FLIP)
parser.add_argument('--warp_sample', type=int, default=config.WARP_SAMPLE) 
parser.add_argument('--cache_features', type=int, default=config.CACHE_FEATURES) 
parser.add_argument('--sh_skips', type=int, default=config.SH_SKIPS)
parser.add_argument('--concat_uv_diff', type=int, default=config.CONCAT_UV_DIFF)
parser.add_argument('--sh_pose', type=int, default=config.SH_POSE)
parser.add_argument('--use_mlp', type=int, default=config.USE_MLP) 
parser.add_argument('--skip_cache_dim', type=int, default=config.SKIP_CACHE_DIM) 
parser.add_argument('--job_id', type=int, default=config.JOB_ID)
parser.add_argument('--use_exp', type=int, default=config.USE_EXP)
parser.add_argument('--c_selection', type=int, default=config.C_SELECTION)
parser.add_argument('--float_image', type=int, default=config.FLOAT_IMAGE)
args = parser.parse_args()

def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 5:
        lr = original_lr * 0.2 * epoch
    elif epoch < 50:
        lr = original_lr
    elif epoch < 100:
        lr = 0.1 * original_lr
    else:
        lr = 0.01 * original_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    """ADD TORCH MANUAL SEED"""
    torch.manual_seed(0) 
    # Fixed since we are training the DNR++ Baseline
    args.single_temporal = True
    args.use_warp = False
    args.cache_features = False

    stats = util.DATASET_STATS[args.dataset_name]
    stats = stats.split(',')
    stats = [float(x) for x in stats]
    mean = stats[:3]
    std = stats[3:]

    logger = Logger(args, baseline=True)
    float_image = args.float_image
        
    dataset = UVDataset(args.data, args.croph, args.cropw, args.train, args.view_direction, \
                        args.useAug, use_mask=args.use_mask,  \
                        single_temporal=args.single_temporal, mean=mean, std=std, float_image=float_image)  

    
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    device = torch.device("cuda")

    if args.load:
        print('Loading Saved Model')
        model = torch.load(os.path.join(args.checkpoint, args.load))
        step = args.load_step
    else:
        model = PipeLine(args.texturew, args.textureh, args.texture_dim, \
                         args.use_pyramid, args.view_direction, args.use_lowpass, \
                         args.use_upconv, args.low_pass_sigma, device, \
                         use_instance_norm=args.use_instance_norm, use_warp=args.use_warp, \
                         network_size=args.network_size, predict_mask=args.predict_mask, cache_layer=args.cache_layer, concat_uv_diff=args.concat_uv_diff,
                         use_pose=args.use_pose, warp_sample=args.warp_sample, cache_features=args.cache_features, 
                         sh_skips=args.sh_skips, sh_pose=args.sh_pose, use_mlp=args.use_mlp, skip_cache_dim=args.skip_cache_dim,
                         use_exp=args.use_exp, single_temporal=args.single_temporal)
        step = 0

    l2 = args.l2.split(',')
    l2 = [float(x) for x in l2]
    betas = args.betas.split(',')
    betas = [float(x) for x in betas]
    betas = tuple(betas)
    unet_weight_decay = args.unet_decay

    lowest_validation = {
        "val_l1": 999,
        "val_psnr": -999,
        "val_ssim": -999
    }

    optimizer = Adam([
        {'params': [model.texture.textures[i].layer1 for i in range(16)], 'weight_decay': l2[0], 'lr': args.lr},
        {'params': [model.texture.textures[i].layer2 for i in range(16)], 'weight_decay': l2[1], 'lr': args.lr},
        {'params': [model.texture.textures[i].layer3 for i in range(16)], 'weight_decay': l2[2], 'lr': args.lr},
        {'params': [model.texture.textures[i].layer4 for i in range(16)], 'weight_decay': l2[3], 'lr': args.lr},
        {'params': model.unet.parameters(), 'lr': 0.1 * args.lr, 'weight_decay':unet_weight_decay},
        ],
        betas=betas, eps=args.eps)
        
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()
    torch.set_grad_enabled(True)

    # SETUP MODEL IN WEIGHTS AND BIAS
    if not args.use_tensorboard:
        logger.set_wandb_watch(model)

    criterion = nn.L1Loss()

    # set up perceptual loss:
    perceptual_loss = VGGPerceptualLoss()
    perceptual_loss = torch.nn.DataParallel(perceptual_loss)
    perceptual_loss.to(device)
    
    total_iterations = 0 # used for tensorboard logging of images
    image_update = 1000 # how often to save image training progress
    do_validation = 1 # how often to perform validation 

    print('Training started')
    for e in range(1, 1+args.epoch):
        print('Epoch {}'.format(e))
        model.train()

        if args.use_lr_decay:
            adjust_learning_rate(optimizer, e, args.lr)

        for n, samples in enumerate(dataloader):
            
            """
            FORWARD PASSS - for the neural network
            """
            if args.view_direction: 
                # DATA PROCESSING   
                images = samples['img']
                uv_maps = samples['uv_map']
                img_mask = samples['img_mask']
                background = samples['bg']
                extrinsics = samples['extrinsics']
                scale = 1
                
                step += images.shape[0]
                optimizer.zero_grad()

                # FORWARD PASS
                RGB_texture, preds_rgb, preds_mask = model(uv_maps.to(device), extrinsics.to(device), device, scale, background) # background isn't used
                preds = preds_mask * preds_rgb + (1. - preds_mask) * background.to(device)

            
            """
            LOSSES - for Training
            """
            # Neural Texture Loss
            texture_loss =args.nt_coef * criterion(RGB_texture.cpu()*img_mask, images*img_mask)

            # L1 Losses
            l1_loss =  criterion(preds.cpu(), images)

            # do perceptual loss
            feature_loss = 0.1 * perceptual_loss(util.denormalize_img(preds.cpu().permute(0,2,3,1), mean=mean, std=std).permute(0,3,1,2).to(device), util.denormalize_img(images.cpu().permute(0,2,3,1), mean=mean, std=std).permute(0,3,1,2).to(device))
            feature_loss = feature_loss.mean()

            loss = texture_loss + l1_loss + feature_loss
            loss.backward()

            optimizer.step()

            if total_iterations % image_update//5 == 0:
                logger.log_baseline_losses(total_iterations, args, texture_loss, l1_loss, feature_loss, loss)
            if total_iterations % image_update == 0:
                logger.log_baseline_images(args, total_iterations, model, images, RGB_texture, preds, preds_rgb, preds_mask, img_mask, background, mean, std)
                
            """ COMMAND LINE LOGGING """
            print('loss at total iter {} / step {}: {}'.format(total_iterations, step, loss.item()))
            total_iterations += 1

        """
        VALIDATION
        """
        if e % do_validation == 0:
            print("Doing Validation")
            val_dict = validation_baseline(model, args, validation=args.validation, logger=logger, epoch=e)
            model.train()
            # check whether or not to save the model
            compare_val = {
                'val_l1': val_dict['val_l1'] < lowest_validation['val_l1'],
                'val_psnr': val_dict['val_psnr'] > lowest_validation['val_psnr'],
                'val_ssim': val_dict['val_ssim'] > lowest_validation['val_ssim']
            }

            for k, key in enumerate(val_dict.keys()):
                if compare_val[key]:
                    """SAVE MODEL AND OPTIMIZER"""
                    lowest_validation[key] = val_dict[key]
                    training_file = os.path.join(logger.checkpoint_dir, "best_{}.tar".format(key))
                    torch.save({
                                'epoch': e,
                                'dataset': args.data,
                                'batch_size': args.batch,
                                'current_validation': val_dict,
                                'lowest_validation_loss':lowest_validation,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                    }, training_file)
        
            logger.log_validation_losses(lowest_validation, e)
                
        """
        CHECKPOINT LOGGING
        """
        if e % args.epoch_per_checkpoint == 0:
            print('Saving Checkpoint')
            training_file = os.path.join(logger.checkpoint_dir, "model_epoch-{}.tar".format(e))
            torch.save({
                        'epoch': e,
                        'dataset': args.data,
                        'batch_size': args.batch,
                        'current_validation': val_dict,
                        'lowest_validation_loss':lowest_validation,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
            }, training_file)

    if args.use_tensorboard:
        logger.close_writer()

    print('Saving Last Checkpoint')
    torch.save(model, os.path.join(logger.checkpoint_dir, 'epoch_final_{}.pt'.format(e)))

if __name__ == '__main__':
    main()
