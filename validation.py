import torch
from torch.utils.data import DataLoader
from dataset.warp_uv_dataset import WarpUVDataset
from torchvision.utils import make_grid
from dataset.uv_dataset import UVDataset
import util
import losses
import wandb

def validation_warp(model, args, validation="validation", logger=None, epoch=0):
    # define the dataloader:
    use_aug = False

    stats = util.DATASET_STATS[args.dataset_name]
    stats = stats.split(',')
    stats = [float(x) for x in stats]
    mean = stats[:3]
    std = stats[3:]
    croph, cropw = 0, 0 # DONT random crop
    float_image = args.float_image

    dataset = WarpUVDataset(args.data, H=croph, W=cropw, split=validation, view_direction=args.view_direction,\
                            use_aug=use_aug, use_mask=True, t_diff=1,
                            mean=mean, std=std, use_pose=args.use_pose, normSplit=args.train, use_exp=args.use_exp, float_image=float_image)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    device = torch.device("cuda:{}".format(args.cuda))

    L1 = torch.nn.L1Loss()

    validation_L1_1 = 0
    validation_PSNR_1 = 0
    validation_SSIM_1 = 0

    validation_L1_2 = 0
    validation_PSNR_2 = 0
    validation_SSIM_2 = 0

    model.eval()
    with torch.no_grad():
        print('Validation started')
        for n, samples in enumerate(dataloader):
        
            if args.view_direction:    
                images = samples['img']
                uv_maps = samples['uv_map']
                background = samples['bg']
                extrinsics = samples['extrinsics']

                if args.use_pose:
                    pose = samples['pose']

                if args.use_exp:
                    exp = samples['exp']

                scale = 1

                if args.predict_mask:
                    if args.use_pose:
                        if args.use_exp:
                            _, pred_1_rgb, pred_1_mask, pred_2_rgb, pred_2_mask, _ = model(uv_maps.to(device), extrinsics.to(device), device, scale, background.to(device), pose.to(device), exp.to(device))
                        else:
                            _, pred_1_rgb, pred_1_mask, pred_2_rgb, pred_2_mask, _ = model(uv_maps.to(device), extrinsics.to(device), device, scale, background.to(device), pose.to(device))
                    else:
                        _, pred_1_rgb, pred_1_mask, pred_2_rgb, pred_2_mask, _ = model(uv_maps.to(device), extrinsics.to(device), device, scale, background.to(device))

                    pred_1 = pred_1_mask * pred_1_rgb + (1. - pred_1_mask) * background[:,0,...].to(device)
                    pred_2 = pred_2_mask * pred_2_rgb + (1. - pred_2_mask) * background[:,1,...].to(device)
                else:
                    _, pred_1, pred_2, _ = model(uv_maps.to(device), extrinsics.to(device), device, scale, background.to(device))
            
            batch_l1_1 =            L1(torch.clip(util.denormalize_img(pred_1.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images[:,0,...].permute(0,2,3,1),mean=mean, std=std),min=0, max=1))
            batch_psnr_1 = losses.PSNR(torch.clip(util.denormalize_img(pred_1.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images[:,0,...].permute(0,2,3,1),mean=mean, std=std),min=0, max=1))
            batch_ssim_1 = losses.SSIM(torch.clip(util.denormalize_img(pred_1.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images[:,0,...].permute(0,2,3,1),mean=mean, std=std),min=0, max=1))

            batch_l1_2 =            L1(torch.clip(util.denormalize_img(pred_2.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images[:,1,...].permute(0,2,3,1),mean=mean, std=std),min=0, max=1))
            batch_psnr_2 = losses.PSNR(torch.clip(util.denormalize_img(pred_2.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images[:,1,...].permute(0,2,3,1),mean=mean, std=std),min=0, max=1))
            batch_ssim_2 = losses.SSIM(torch.clip(util.denormalize_img(pred_2.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images[:,1,...].permute(0,2,3,1),mean=mean, std=std),min=0, max=1))

            # update validation 
            validation_L1_1 += batch_l1_1
            validation_PSNR_1 += batch_psnr_1
            validation_SSIM_1 += batch_ssim_1

            validation_L1_2 += batch_l1_2
            validation_PSNR_2 += batch_psnr_2
            validation_SSIM_2 += batch_ssim_2

    # Find average validation metrics
    validation_L1_1 /= len(dataloader)
    validation_PSNR_1 /= len(dataloader)
    validation_SSIM_1 /= len(dataloader)

    validation_L1_2 /= len(dataloader)
    validation_PSNR_2 /= len(dataloader)
    validation_SSIM_2 /= len(dataloader)

    # Plotting and Logging
    print('Logging Validation Losses and Images')
    logger.log_validation_losses_loop(args, validation_L1_1, validation_L1_2,
                              validation_PSNR_1, validation_PSNR_2,
                              validation_SSIM_1, validation_SSIM_2,
                              epoch)

    
    logger.log_validation_images(args, images, pred_1, pred_2, 
                                 pred_1_rgb, pred_1_mask, 
                                 pred_2_rgb, pred_2_mask, 
                                 mean, std, epoch)


    model.train()

    validation_L1 =  validation_L1_2.item()
    validation_PSNR =  validation_PSNR_2.item()
    validation_SSIM =  validation_SSIM_2.item()

    return {
        "val_l1": validation_L1,
        "val_psnr": validation_PSNR,
        "val_ssim": validation_SSIM
    }

def validation_baseline(model, args, validation="validation", logger=None, epoch=0):
    stats = util.DATASET_STATS[args.dataset_name]
    stats = stats.split(',')
    stats = [float(x) for x in stats]
    mean = stats[:3]
    std = stats[3:]
    # define the dataloader:
    use_aug = False
    croph, cropw = 0, 0 # DONT random crop
    float_image = args.float_image
    dataset = UVDataset(args.data, H=croph, W=cropw, split=validation, view_direction=args.view_direction,\
                        use_aug=use_aug, use_mask=True,
                        single_temporal=args.single_temporal,
                        mean=mean, std=std, float_image=float_image)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    device = torch.device("cuda:{}".format(args.cuda))

    L1 = torch.nn.L1Loss()

    validation_L1 = 0
    validation_PSNR = 0
    validation_SSIM = 0

    with torch.no_grad():
        print('Validation started')
        for n, samples in enumerate(dataloader):
            if args.view_direction:         
                images = samples['img']
                uv_maps = samples['uv_map']
                background = samples['bg']
                extrinsics = samples['extrinsics']

                scale = 1
                if args.predict_mask:
                    _, preds_rgb, preds_mask = model(uv_maps.to(device), extrinsics.to(device), device, scale, background) # background isn't used
                    preds = preds_mask * preds_rgb + (1. - preds_mask) * background.to(device)
                else:
                    _, preds = model(uv_maps.to(device), extrinsics.to(device), device, scale, background.to(device))
            

            batch_l1 =            L1(torch.clip(util.denormalize_img(preds.cpu().permute(0,2,3,1), mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images.permute(0,2,3,1), mean=mean, std=std),min=0, max=1))
            batch_psnr = losses.PSNR(torch.clip(util.denormalize_img(preds.cpu().permute(0,2,3,1), mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images.permute(0,2,3,1), mean=mean, std=std),min=0, max=1))
            batch_ssim = losses.SSIM(torch.clip(util.denormalize_img(preds.cpu().permute(0,2,3,1), mean=mean, std=std),min=0, max=1), torch.clip(util.denormalize_img(images.permute(0,2,3,1), mean=mean, std=std),min=0, max=1))

            validation_L1 += batch_l1
            validation_PSNR += batch_psnr
            validation_SSIM += batch_ssim

    # Find average validation metrics
    validation_L1 /= len(dataloader)
    validation_PSNR /= len(dataloader)
    validation_SSIM /= len(dataloader)

    # Validation Plotting
    """ WEIGHTS AND BIAS LOGGING """
    if args.use_tensorboard:
        logger.writer.add_scalar("Validation Photometric (L1)", validation_L1.item(), epoch)
        logger.writer.add_scalar("Validation PSNR", validation_PSNR.item(), epoch)
        logger.writer.add_scalar("Validation SSIM", validation_SSIM.item(), epoch)
    else:
        wandb.log({"Validation Photometric (L1)": validation_L1})
        wandb.log({"Validation PSNR": validation_PSNR})
        wandb.log({"Validation SSIM": validation_SSIM})

    display_images = util.denormalize_img(images.cpu().permute(0,2,3,1), mean=mean, std=std).permute(0,3,1,2)
    display_images = make_grid(display_images, nrow=display_images.shape[0]//2)

    clip_preds = torch.clip(util.denormalize_img(preds.cpu().permute(0,2,3,1), mean=mean, std=std),min=0, max=1).permute(0,3,1,2)
    clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2)

    if args.use_tensorboard:
        logger.writer.add_figure("Validation/GT", util.show(display_images), epoch, close=True)
        logger.writer.add_figure('Validation/Predictions-clip_values', util.show(clip_preds), epoch, close=True)
    else:
        wandb_image = wandb.Image(display_images, caption="GT Images")          
        wandb.log({"Validation/GT": wandb_image})
        wandb_image = wandb.Image(clip_preds, caption="Predictions (clip preds + train norm)")          
        wandb.log({"Validation/Predictions-clip_values": wandb_image})

    if args.predict_mask:
        display_images = torch.clip(util.denormalize_img(preds_rgb.cpu().permute(0,2,3,1), mean=mean, std=std), min=0, max=1).permute(0,3,1,2)
        display_images = make_grid(display_images, nrow=display_images.shape[0]//2, normalize=False)
        clip_preds = torch.clip(preds_mask.cpu().permute(0,2,3,1),min=0, max=1).permute(0,3,1,2)
        clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2, normalize=False,)

        if args.use_tensorboard:
            logger.writer.add_figure("Validation/RGB Prediction", util.show(display_images), epoch, close=True)
            logger.writer.add_figure('Validation/Mask Prediction', util.show(clip_preds), epoch, close=True)
        else:
            wandb_image = wandb.Image(clip_preds, caption="Mask Prediction")          
            wandb.log({"Validation/Mask Prediction": wandb_image})
            wandb_image = wandb.Image(display_images, caption="RGB Prediction")          
            wandb.log({"Validation/RGB Prediction": wandb_image})

    model.train()

    return {
        "val_l1": validation_L1.item(),
        "val_psnr": validation_PSNR.item(),
        "val_ssim": validation_SSIM.item()
    }
