import wandb
import os
from torch.utils.tensorboard import SummaryWriter
import json
import util
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

def test():
    return

class Logger():
    def __init__(self, args, baseline=False) -> None:
        self.args = args
        if baseline:
            self.project_name = 'WACV2023-Baseline'
        else:
            self.project_name = 'WACV2023-Warp'
        self.use_tensorboard = args.use_tensorboard

        if args.use_tensorboard == False:
            wandb.init(project=self.project_name, config=args)
            self.run_name = wandb.run.name

        create_log = True
        count = 0
        while create_log:
            count += 1

            if args.use_tensorboard:
                self.run_name = "model-{}".format(str(args.job_id), str(count))
            local_path = './logs'
            
            log_dir = os.path.join(local_path, self.project_name, self.run_name)
            self.checkpoint_dir = os.path.join(local_path, self.project_name, self.run_name, 'checkpoints')

            if os.path.exists(log_dir) is False:
                os.makedirs(log_dir)
                os.makedirs(self.checkpoint_dir)

                if args.use_tensorboard:
                    self.writer = SummaryWriter(log_dir=log_dir)
                else:
                    self.writer = None

                # save all configs in the run folder
                with open(os.path.join(log_dir, 'commandline_args.txt'), 'w') as f:
                    json.dump(args.__dict__, f, indent=2)

                create_log = False
    
    def close_writer(self):
        self.writer.close()

    def set_wandb_watch(self, model, freq=100):
        wandb.watch(model, log_freq=freq)


    def log_training_losses(self, step, texture_loss, l1_1, l1_2, feature_loss_1, feature_loss_2, loss):
        # LOG THE LOSS GRAPHS
        if not self.use_tensorboard:
            """ WEIGHTS AND BIAS LOGGING """
            wandb.log({"RGB Texture Loss": texture_loss})
            wandb.log({"Predicted t=1 Loss": l1_1})
            wandb.log({"Predicted t=2 Loss": l1_2})
            wandb.log({"Perceptual Loss t=1": feature_loss_1})
            wandb.log({"Perceptual Loss t=2": feature_loss_2})
            wandb.log({"Total Loss": loss})
        else:
            """ TENSORBOARD LOGGING """
            print('Updating Tensorboard')
            self.writer.add_scalar('train/RGB Texture Loss', texture_loss.item(), step)
            self.writer.add_scalar('train/Predicted t=1 Loss', l1_1.item(), step)
            self.writer.add_scalar('train/Predicted t=2 Loss', l1_2.item(), step)

            self.writer.add_scalar('train/Perceptual Loss t=1', feature_loss_1.item(), step)
            self.writer.add_scalar('train/Perceptual Loss t=2', feature_loss_2.item(), step)
            self.writer.add_scalar('train/Total Loss', loss.item(), step)

    def log_validation_losses(self, lowest_validation, e):
        # LOG THE BEST VALIDAION POINTS
        if self.use_tensorboard:
            self.writer.add_scalar('Best Val L1', lowest_validation['val_l1'], e)
            self.writer.add_scalar('Best Val PSNR', lowest_validation['val_psnr'], e)
            self.writer.add_scalar('Best Val SSIM', lowest_validation['val_ssim'], e)
        else:
            wandb.log({"Best Val L1": lowest_validation['val_l1']})
            wandb.log({"Best Val PSNR": lowest_validation['val_psnr']})
            wandb.log({"Best Val SSIM": lowest_validation['val_ssim']})

    
    def log_training_images(self, step, model, images, img_mask, RGB_texture, pred_1, pred_2, pred_1_rgb, pred_1_mask, pred_2_rgb, pred_2_mask, mean, std):
        images = images.detach().cpu()
        RGB_texture = RGB_texture.detach().cpu()
        pred_1 = pred_1.detach().cpu()
        pred_2 = pred_2.detach().cpu()
        if self.args.predict_mask:
            pred_1_rgb = pred_1_rgb.detach().cpu()
            pred_1_mask = pred_1_mask.detach().cpu()
            pred_2_rgb = pred_2_rgb.detach().cpu()
            pred_2_mask = pred_2_mask.detach().cpu()

        preds = torch.cat((pred_1.unsqueeze(1), pred_2.unsqueeze(1)), dim=1)

        # visual batch training results (t=1)
        display_images = util.denormalize_img(images[:,0,...].cpu().permute(0,2,3,1),mean=mean, std=std).permute(0,3,1,2)
        difference_display = display_images.clone()

        display_images = make_grid(display_images, nrow=display_images.shape[0]//2)
        clip_preds = torch.clip(util.denormalize_img(pred_1.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1).permute(0,3,1,2)
        difference_clip_preds = clip_preds.clone()
        clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2)

        if self.use_tensorboard:
            self.writer.add_figure("Training/GT (t=1)", util.show(display_images), step, close=True)
            self.writer.add_figure('Training/Predictions-clip_values (t=1)', util.show(clip_preds), step, close=True)
        else:
            wandb_image = wandb.Image(display_images, caption="GT Images (t=1)")          
            wandb.log({"Training/GT (t=1)": wandb_image})
            wandb_image = wandb.Image(clip_preds, caption="Predictions (clip preds + train norm)")          
            wandb.log({"Training/Predictions-clip_values (t=1)": wandb_image})

        # t=2
        display_images = util.denormalize_img(images[:,1,...].cpu().permute(0,2,3,1),mean=mean, std=std).permute(0,3,1,2)
        difference_display_2 = torch.clone(display_images)

        display_images = make_grid(display_images, nrow=display_images.shape[0]//2)
        clip_preds = torch.clip(util.denormalize_img(pred_2.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1).permute(0,3,1,2)
        difference_clip_preds_2 = clip_preds.clone()
        clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2)

        if self.use_tensorboard:
            self.writer.add_figure("Training/GT (t=2)", util.show(display_images), step, close=True)
            self.writer.add_figure('Training/Predictions-clip_values (t=2)', util.show(clip_preds), step, close=True)
        else:
            wandb_image = wandb.Image(display_images, caption="GT Images (t=2)")          
            wandb.log({"Training/GT (t=2)": wandb_image})
            wandb_image = wandb.Image(clip_preds, caption="Predictions (clip preds + train norm)")          
            wandb.log({"Training/Predictions-clip_values (t=2)": wandb_image})


        ddd = make_grid(difference_display - difference_display_2, nrow=difference_display.shape[0]//2)
        if self.use_tensorboard:
            self.writer.add_figure("Training/GT Difference Map", util.show((ddd-torch.min(ddd))/torch.max(ddd)), step, close=True)
        else:  
            wandb_image = wandb.Image(ddd, caption="GT Difference Map")       
            wandb.log({"Training/GT Difference Map": wandb_image})

        dcp = make_grid(difference_clip_preds - difference_clip_preds_2, nrow=difference_clip_preds_2.shape[0]//2)    
        if self.use_tensorboard:
            self.writer.add_figure("Training/Pred Difference Map", util.show((dcp-torch.min(dcp))/torch.max(dcp)), step, close=True)  
        else:      
            wandb_image = wandb.Image(dcp, caption="Pred Difference Map") 
            wandb.log({"Training/Pred Difference Map": wandb_image})

        if self.args.predict_mask:
            # t=1
            display_images = torch.clip(util.denormalize_img(pred_1_rgb.cpu().permute(0,2,3,1),mean=mean, std=std), min=0, max=1).permute(0,3,1,2)
            display_images = make_grid(display_images, nrow=display_images.shape[0]//2, normalize=False)
            clip_preds = torch.clip(pred_1_mask.cpu().permute(0,2,3,1),min=0, max=1).permute(0,3,1,2)
            clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2, normalize=False,)

            if self.use_tensorboard:
                self.writer.add_figure('Training/Mask Prediction (t=1)', util.show(clip_preds), step, close=True)
                self.writer.add_figure("Training/RGB Prediction (t=1)", util.show(display_images), step, close=True)
            else:
                wandb_image = wandb.Image(clip_preds, caption="Mask Prediction (t=1)")          
                wandb.log({"Training/Mask Prediction (t=1)": wandb_image})
                wandb_image = wandb.Image(display_images, caption="RGB Prediction (t=1)")          
                wandb.log({"Training/RGB Prediction (t=1)": wandb_image})

            # t=2
            display_images = torch.clip(util.denormalize_img(pred_2_rgb.cpu().permute(0,2,3,1),mean=mean, std=std), min=0, max=1).permute(0,3,1,2)
            display_images = make_grid(display_images, nrow=display_images.shape[0]//2, normalize=False)
            clip_preds = torch.clip(pred_2_mask.cpu().permute(0,2,3,1),min=0, max=1).permute(0,3,1,2)
            clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2, normalize=False,)
            
            if self.use_tensorboard:
                self.writer.add_figure('Training/Mask Prediction (t=2)', util.show(clip_preds), step, close=True)
                self.writer.add_figure("Training/RGB Prediction (t=2)", util.show(display_images), step, close=True)
            else:
                wandb_image = wandb.Image(clip_preds, caption="Mask Prediction (t=2)")          
                wandb.log({"Training/Mask Prediction (t=2)": wandb_image})
                wandb_image = wandb.Image(display_images, caption="RGB Prediction (t=2)")          
                wandb.log({"Training/RGB Prediction (t=2)": wandb_image})

        
        # Visualize the Neural Texture Layers
        if self.args.use_pyramid:
            fig = plt.figure(figsize=(25,10))
            nt_layer1 = []
            nt_layer2 = []
            nt_layer3 = []
            nt_layer4 = []

            for dim in range(3):
                nt_layer1.append(model.module.texture.layer1[dim].clone().detach().cpu())
                nt_layer2.append(model.module.texture.layer2[dim].clone().detach().cpu())
                nt_layer3.append(model.module.texture.layer3[dim].clone().detach().cpu())
                nt_layer4.append(model.module.texture.layer4[dim].clone().detach().cpu())
            nt_layer1 = torch.stack(nt_layer1).squeeze().unsqueeze(0)
            nt_layer2 = torch.stack(nt_layer2).squeeze().unsqueeze(0)
            nt_layer3 = torch.stack(nt_layer3).squeeze().unsqueeze(0)
            nt_layer4 = torch.stack(nt_layer4).squeeze().unsqueeze(0)

            nt_layer2_s = F.interpolate(nt_layer2, scale_factor=2, mode='bilinear')
            nt_layer3_s = F.interpolate(nt_layer3, scale_factor=4, mode='bilinear')
            nt_layer4_s = F.interpolate(nt_layer4, scale_factor=8, mode='bilinear')
            nt_total = nt_layer1 + nt_layer2_s + nt_layer3_s + nt_layer4_s

            ax = plt.subplot2grid((2,5), (0,0))
            ax.imshow(util.denormalize_img(nt_layer1.squeeze().permute(1,2,0),mean=mean, std=std).numpy())
            ax.title.set_text('Neural Texture Layer 1 (dim1-3)')
            ax1 = plt.subplot2grid((2,5), (0,1))
            ax1.imshow(util.denormalize_img(nt_layer2.squeeze().permute(1,2,0),mean=mean, std=std).numpy())
            ax1.title.set_text('Neural Texture Layer 2 (dim1-3)')
            ax2 = plt.subplot2grid((2,5), (0,2))
            ax2.imshow(util.denormalize_img(nt_layer3.squeeze().permute(1,2,0),mean=mean, std=std).numpy())
            ax2.title.set_text('Neural Texture Layer 3 (dim1-3)')
            ax3 = plt.subplot2grid((2,5), (0,3))
            ax3.imshow(util.denormalize_img(nt_layer4.squeeze().permute(1,2,0),mean=mean, std=std).numpy())
            ax3.title.set_text('Neural Texture Layer 4 (dim1-3)')
            ax4 = plt.subplot2grid((2,5), (0,4))
            ax4.imshow(util.denormalize_img(nt_total.squeeze().permute(1,2,0),mean=mean, std=std).numpy())
            ax4.title.set_text('Combined Neural Texture')

            # Visualize if there are any negative values in the Neural Texture
            n_nt_layer1 = nt_layer1 * 1 
            n_nt_layer2 = nt_layer2 * 1 
            n_nt_layer3 = nt_layer3 * 1 
            n_nt_layer4 = nt_layer4 * 1
            n_nt_total = nt_total * 1

            ax5 = plt.subplot2grid((2,5), (1,0))
            ax5.imshow(util.denormalize_img(n_nt_layer1.squeeze().permute(1,2,0),mean=mean, std=std).numpy() * -1)
            ax5.title.set_text('Negative Layer 1 (dim1-3)')
            ax6 = plt.subplot2grid((2,5), (1,1))
            ax6.imshow(util.denormalize_img(n_nt_layer2.squeeze().permute(1,2,0),mean=mean, std=std).numpy() * -1)
            ax6.title.set_text('Negative Layer 2 (dim1-3)')
            ax7 = plt.subplot2grid((2,5), (1,2))
            ax7.imshow(util.denormalize_img(n_nt_layer3.squeeze().permute(1,2,0),mean=mean, std=std).numpy() * -1)
            ax7.title.set_text('Negative Layer 3 (dim1-3)')
            ax8 = plt.subplot2grid((2,5), (1,3))
            ax8.imshow(util.denormalize_img(n_nt_layer4.squeeze().permute(1,2,0),mean=mean, std=std).numpy() * -1)
            ax8.title.set_text('Negative Layer 4 (dim1-3)')
            ax9 = plt.subplot2grid((2,5), (1,4))
            ax9.imshow(util.denormalize_img(n_nt_total.squeeze().permute(1,2,0),mean=mean, std=std).numpy() * -1)
            ax9.title.set_text('Combined Negative Texture')

        else:
            nt_layer1 = []
            fig = plt.figure(figsize=(10,10))
            for dim in range(3):
                nt_layer1.append(model.module.texture.layer1[dim].clone().detach().cpu())

            nt_layer1 = torch.stack(nt_layer1).squeeze().unsqueeze(0)

            ax = plt.subplot2grid((2,1), (0,0))
            ax.imshow(util.denormalize_img(nt_layer1.squeeze().permute(1,2,0),mean=mean, std=std).numpy())
            ax.title.set_text('Neural Texture Layer 1 (dim1-3)')
            
            # Visualize if there are any negative values in the Neural Texture
            n_nt_layer1 = nt_layer1 * 1 

            ax5 = plt.subplot2grid((2,1), (1,0))
            ax5.imshow(util.denormalize_img(n_nt_layer1.squeeze().permute(1,2,0),mean=mean, std=std).numpy() * -1)
            ax5.title.set_text('Negative Layer 1 (dim1-3)')

        if self.use_tensorboard:
            self.writer.add_figure('Neural Texture', fig, step, close=True)
        else:
            fig.canvas.draw()
            wandb_plot_b = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            wandb_plot_b = wandb_plot_b.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            wandb.log({"Neural Texture": wandb.Image(wandb_plot_b)})


        # Visualize the Last 3 Neural Texture Layers
        if self.args.use_pyramid:
            fig = plt.figure(figsize=(25,10))
            nt_layer1 = []
            nt_layer2 = []
            nt_layer3 = []
            nt_layer4 = []

            for dim in range(13,16):
                nt_layer1.append(model.module.texture.layer1[dim].clone().detach().cpu())
                nt_layer2.append(model.module.texture.layer2[dim].clone().detach().cpu())
                nt_layer3.append(model.module.texture.layer3[dim].clone().detach().cpu())
                nt_layer4.append(model.module.texture.layer4[dim].clone().detach().cpu())
            nt_layer1 = torch.stack(nt_layer1).squeeze().unsqueeze(0)
            nt_layer2 = torch.stack(nt_layer2).squeeze().unsqueeze(0)
            nt_layer3 = torch.stack(nt_layer3).squeeze().unsqueeze(0)
            nt_layer4 = torch.stack(nt_layer4).squeeze().unsqueeze(0)

            nt_layer2_s = F.interpolate(nt_layer2, scale_factor=2, mode='bilinear')
            nt_layer3_s = F.interpolate(nt_layer3, scale_factor=4, mode='bilinear')
            nt_layer4_s = F.interpolate(nt_layer4, scale_factor=8, mode='bilinear')
            nt_total = nt_layer1 + nt_layer2_s + nt_layer3_s + nt_layer4_s

            ax = plt.subplot2grid((2,5), (0,0))
            ax.imshow(nt_layer1.squeeze().permute(1,2,0).numpy())
            ax.title.set_text('Neural Texture Layer 1 (dim1-3)')
            ax1 = plt.subplot2grid((2,5), (0,1))
            ax1.imshow(nt_layer2.squeeze().permute(1,2,0).numpy())
            ax1.title.set_text('Neural Texture Layer 2 (dim1-3)')
            ax2 = plt.subplot2grid((2,5), (0,2))
            ax2.imshow(nt_layer3.squeeze().permute(1,2,0).numpy())
            ax2.title.set_text('Neural Texture Layer 3 (dim1-3)')
            ax3 = plt.subplot2grid((2,5), (0,3))
            ax3.imshow(nt_layer4.squeeze().permute(1,2,0).numpy())
            ax3.title.set_text('Neural Texture Layer 4 (dim1-3)')
            ax4 = plt.subplot2grid((2,5), (0,4))
            ax4.imshow(nt_total.squeeze().permute(1,2,0).numpy())
            ax4.title.set_text('Combined Neural Texture')

            # Visualize if there are any negative values in the Neural Texture
            n_nt_layer1 = nt_layer1 * 1 
            n_nt_layer2 = nt_layer2 * 1 
            n_nt_layer3 = nt_layer3 * 1 
            n_nt_layer4 = nt_layer4 * 1
            n_nt_total = nt_total * 1

            ax5 = plt.subplot2grid((2,5), (1,0))
            ax5.imshow(n_nt_layer1.squeeze().permute(1,2,0).numpy() * -1)
            ax5.title.set_text('Negative Layer 1 (dim1-3)')
            ax6 = plt.subplot2grid((2,5), (1,1))
            ax6.imshow(n_nt_layer2.squeeze().permute(1,2,0).numpy() * -1)
            ax6.title.set_text('Negative Layer 2 (dim1-3)')
            ax7 = plt.subplot2grid((2,5), (1,2))
            ax7.imshow(n_nt_layer3.squeeze().permute(1,2,0).numpy() * -1)
            ax7.title.set_text('Negative Layer 3 (dim1-3)')
            ax8 = plt.subplot2grid((2,5), (1,3))
            ax8.imshow(n_nt_layer4.squeeze().permute(1,2,0).numpy() * -1)
            ax8.title.set_text('Negative Layer 4 (dim1-3)')
            ax9 = plt.subplot2grid((2,5), (1,4))
            ax9.imshow(n_nt_total.squeeze().permute(1,2,0).numpy() * -1)
            ax9.title.set_text('Combined Negative Texture')

            if self.use_tensorboard:
                self.writer.add_figure('Neural Texture (Dim 13-15)', fig, step, close=True)
            else:
                fig.canvas.draw()
                wandb_plot_b = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                wandb_plot_b = wandb_plot_b.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                wandb.log({"Neural Texture (Dim 13-15)": wandb.Image(wandb_plot_b)})

        # Visualize the inputs and outputs of the neural renderer
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(241)
        ax.imshow(util.denormalize_img(images[0,0,...].permute(1,2,0),mean=mean, std=std).numpy())
        ax.title.set_text('Input Image (t=1)')
        ax2 = fig.add_subplot(242)
        ax2.imshow((util.denormalize_img(RGB_texture[0].permute(1,2,0),mean=mean, std=std)*img_mask[0,0,...].permute(1,2,0)).numpy())
        ax2.title.set_text('NT Image (t=1) w/ Mask')

        ax8 = fig.add_subplot(243)
        ax8.imshow((util.denormalize_img(images[0,0,...].permute(1,2,0),mean=mean, std=std)*img_mask[0,0,...].permute(1,2,0)).numpy())
        ax8.title.set_text('GT Image (t=1) w/ Mask')

        ax5 = fig.add_subplot(245)
        ax5.imshow(util.denormalize_img(images[0,1,...].permute(1,2,0),mean=mean, std=std).numpy())
        ax5.title.set_text('Input Image (t=2)')

        ax7 = fig.add_subplot(246)
        ax7.imshow(util.denormalize_img(pred_1[0].permute(1,2,0),mean=mean, std=std).numpy())
        ax7.title.set_text('Rendered Image (t=1)')

        ax3 = fig.add_subplot(247)
        ax3.imshow(util.denormalize_img(pred_2[0].permute(1,2,0),mean=mean, std=std).numpy())
        ax3.title.set_text('Rendered Image (t=2)')

        if self.use_tensorboard:
                self.writer.add_figure('Training/Outputs', fig, step, close=True)
        else:
            fig.canvas.draw()
            wandb_plot_c = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            wandb_plot_c = wandb_plot_c.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            wandb.log({"Training/Outputs": wandb.Image(wandb_plot_c)})

    def log_baseline_losses(self, step, args, texture_loss, l1_loss, feature_loss, loss):
        """ WEIGHTS AND BIAS LOGGING """
        if not args.use_tensorboard:
            wandb.log({"RGB Texture Loss": texture_loss})
            wandb.log({"Predicted Image Loss": l1_loss})
            wandb.log({"Perceptual Loss": feature_loss})
            wandb.log({"Total Loss": loss})
        else:
            """ TENSORBOARD LOGGING """
            print('Updating Tensorboard')
            self.writer.add_scalar('train/RGB Texture Loss', texture_loss.item(), step)
            self.writer.add_scalar('train/Predicted Image Loss', l1_loss.item(), step)
            self.writer.add_scalar('train/Total Loss', loss.item(), step)
            self.writer.add_scalar('train/Perceptual Loss', feature_loss.item(), step)
    
    def log_baseline_images(self, args, step, model, images, RGB_texture, preds, preds_rgb, preds_mask, img_mask, background, mean, std):
        images = images.detach().cpu()
        RGB_texture = RGB_texture.detach().cpu()
        preds = preds.detach().cpu()
        preds_rgb = preds_rgb.detach().cpu()
        preds_mask = preds_mask.detach().cpu()

        error_map_output = torch.abs(images[0] - preds[0])
        error_map_texture = torch.abs(images[0] - RGB_texture[0])

        # Visualize Error Maps
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(121)
        ax.imshow(error_map_output.permute(1,2,0).numpy())
        ax.title.set_text('Prediction Error Map')
        ax1 = fig.add_subplot(122)
        ax1.imshow(error_map_texture.permute(1,2,0).numpy())
        ax1.title.set_text('Texture Error Map')

        if args.use_tensorboard:
            self.writer.add_figure('Error Maps', fig, step, close=True)
        else:
            fig.canvas.draw()
            wandb_plot_a = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            wandb_plot_a = wandb_plot_a.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            wandb.log({"Error Maps": wandb.Image(wandb_plot_a)})

        # visual batch training results
        display_images = util.denormalize_img(images.cpu().permute(0,2,3,1), mean=mean, std=std).permute(0,3,1,2)
        display_images = make_grid(display_images, nrow=display_images.shape[0]//2)
        clip_preds = torch.clip(util.denormalize_img(preds.cpu().permute(0,2,3,1), mean=mean, std=std),min=0, max=1).permute(0,3,1,2)
        clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2)

        if args.use_tensorboard:
            self.writer.add_figure("Training/GT", util.show(display_images), step, close=True)
            self.writer.add_figure('Training/Predictions-clip_values', util.show(clip_preds), step, close=True)
        else:
            wandb_image = wandb.Image(display_images, caption="GT Images")          
            wandb.log({"Training/GT": wandb_image})
            wandb_image = wandb.Image(clip_preds, caption="Predictions (clip preds + train norm)")          
            wandb.log({"Training/Predictions-clip_values": wandb_image})

        if args.predict_mask:
            display_images = torch.clip(util.denormalize_img(preds_rgb.cpu().permute(0,2,3,1), mean=mean, std=std), min=0, max=1).permute(0,3,1,2)
            display_images = make_grid(display_images, nrow=display_images.shape[0]//2, normalize=False)
            clip_preds = torch.clip(preds_mask.cpu().permute(0,2,3,1),min=0, max=1).permute(0,3,1,2)
            clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2, normalize=False,)

            if args.use_tensorboard:
                self.writer.add_figure("Training/RGB Prediction", util.show(display_images), step, close=True)
                self.writer.add_figure('Training/Mask Prediction', util.show(clip_preds), step, close=True)
            else:
                wandb_image = wandb.Image(clip_preds, caption="Mask Prediction")          
                wandb.log({"Training/Mask Prediction": wandb_image})
                wandb_image = wandb.Image(display_images, caption="RGB Prediction")          
                wandb.log({"Training/RGB Prediction": wandb_image})

        # Visualize the Neural Texture Layers
        if args.use_pyramid:
            fig = plt.figure(figsize=(25,10))
            nt_layer1 = []
            nt_layer2 = []
            nt_layer3 = []
            nt_layer4 = []

            for dim in range(3):
                nt_layer1.append(model.module.texture.layer1[dim].clone().detach().cpu())
                nt_layer2.append(model.module.texture.layer2[dim].clone().detach().cpu())
                nt_layer3.append(model.module.texture.layer3[dim].clone().detach().cpu())
                nt_layer4.append(model.module.texture.layer4[dim].clone().detach().cpu())
            nt_layer1 = torch.stack(nt_layer1).squeeze().unsqueeze(0)
            nt_layer2 = torch.stack(nt_layer2).squeeze().unsqueeze(0)
            nt_layer3 = torch.stack(nt_layer3).squeeze().unsqueeze(0)
            nt_layer4 = torch.stack(nt_layer4).squeeze().unsqueeze(0)

            nt_layer2_s = F.interpolate(nt_layer2, scale_factor=2, mode='bilinear')
            nt_layer3_s = F.interpolate(nt_layer3, scale_factor=4, mode='bilinear')
            nt_layer4_s = F.interpolate(nt_layer4, scale_factor=8, mode='bilinear')
            nt_total = nt_layer1 + nt_layer2_s + nt_layer3_s + nt_layer4_s

            ax = plt.subplot2grid((2,5), (0,0))
            ax.imshow(util.denormalize_img(nt_layer1.squeeze().permute(1,2,0), mean=mean, std=std).numpy())
            ax.title.set_text('Neural Texture Layer 1 (dim1-3)')
            ax1 = plt.subplot2grid((2,5), (0,1))
            ax1.imshow(util.denormalize_img(nt_layer2.squeeze().permute(1,2,0), mean=mean, std=std).numpy())
            ax1.title.set_text('Neural Texture Layer 2 (dim1-3)')
            ax2 = plt.subplot2grid((2,5), (0,2))
            ax2.imshow(util.denormalize_img(nt_layer3.squeeze().permute(1,2,0), mean=mean, std=std).numpy())
            ax2.title.set_text('Neural Texture Layer 3 (dim1-3)')
            ax3 = plt.subplot2grid((2,5), (0,3))
            ax3.imshow(util.denormalize_img(nt_layer4.squeeze().permute(1,2,0), mean=mean, std=std).numpy())
            ax3.title.set_text('Neural Texture Layer 4 (dim1-3)')
            ax4 = plt.subplot2grid((2,5), (0,4))
            ax4.imshow(util.denormalize_img(nt_total.squeeze().permute(1,2,0), mean=mean, std=std).numpy())
            ax4.title.set_text('Combined Neural Texture')
            n_nt_layer1 = nt_layer1 * 1 
            n_nt_layer2 = nt_layer2 * 1 
            n_nt_layer3 = nt_layer3 * 1 
            n_nt_layer4 = nt_layer4 * 1
            n_nt_total = nt_total * 1

            ax5 = plt.subplot2grid((2,5), (1,0))
            ax5.imshow(util.denormalize_img(n_nt_layer1.squeeze().permute(1,2,0), mean=mean, std=std).numpy() * -1)
            ax5.title.set_text('Negative Layer 1 (dim1-3)')
            ax6 = plt.subplot2grid((2,5), (1,1))
            ax6.imshow(util.denormalize_img(n_nt_layer2.squeeze().permute(1,2,0), mean=mean, std=std).numpy() * -1)
            ax6.title.set_text('Negative Layer 2 (dim1-3)')
            ax7 = plt.subplot2grid((2,5), (1,2))
            ax7.imshow(util.denormalize_img(n_nt_layer3.squeeze().permute(1,2,0), mean=mean, std=std).numpy() * -1)
            ax7.title.set_text('Negative Layer 3 (dim1-3)')
            ax8 = plt.subplot2grid((2,5), (1,3))
            ax8.imshow(util.denormalize_img(n_nt_layer4.squeeze().permute(1,2,0), mean=mean, std=std).numpy() * -1)
            ax8.title.set_text('Negative Layer 4 (dim1-3)')
            ax9 = plt.subplot2grid((2,5), (1,4))
            ax9.imshow(util.denormalize_img(n_nt_total.squeeze().permute(1,2,0), mean=mean, std=std).numpy() * -1)
            ax9.title.set_text('Combined Negative Texture')
        
        else:
            nt_layer1 = []
            fig = plt.figure(figsize=(10,10))
            for dim in range(3):
                nt_layer1.append(model.module.texture.layer1[dim].clone().detach().cpu())

            nt_layer1 = torch.stack(nt_layer1).squeeze().unsqueeze(0)

            ax = plt.subplot2grid((2,1), (0,0))
            ax.imshow(util.denormalize_img(nt_layer1.squeeze().permute(1,2,0), mean=mean, std=std).numpy())
            ax.title.set_text('Neural Texture Layer 1 (dim1-3)')
            
            # Visualize if there are any negative values in the Neural Texture
            n_nt_layer1 = nt_layer1 * 1 

            ax5 = plt.subplot2grid((2,1), (1,0))
            ax5.imshow(util.denormalize_img(n_nt_layer1.squeeze().permute(1,2,0), mean=mean, std=std).numpy() * -1)
            ax5.title.set_text('Negative Layer 1 (dim1-3)')

        if args.use_tensorboard:
            self.writer.add_figure('Neural Texture', fig, step, close=True)
        else:
            fig.canvas.draw()
            wandb_plot_b = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            wandb_plot_b = wandb_plot_b.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            wandb.log({"Neural Texture": wandb.Image(wandb_plot_b)})

        # Visualize the Last 3 Neural Texture Layers
        if args.use_pyramid:
            fig = plt.figure(figsize=(25,10))
            nt_layer1 = []
            nt_layer2 = []
            nt_layer3 = []
            nt_layer4 = []

            for dim in range(13,16):
                nt_layer1.append(model.module.texture.layer1[dim].clone().detach().cpu())
                nt_layer2.append(model.module.texture.layer2[dim].clone().detach().cpu())
                nt_layer3.append(model.module.texture.layer3[dim].clone().detach().cpu())
                nt_layer4.append(model.module.texture.layer4[dim].clone().detach().cpu())
            nt_layer1 = torch.stack(nt_layer1).squeeze().unsqueeze(0)
            nt_layer2 = torch.stack(nt_layer2).squeeze().unsqueeze(0)
            nt_layer3 = torch.stack(nt_layer3).squeeze().unsqueeze(0)
            nt_layer4 = torch.stack(nt_layer4).squeeze().unsqueeze(0)

            nt_layer2_s = F.interpolate(nt_layer2, scale_factor=2, mode='bilinear')
            nt_layer3_s = F.interpolate(nt_layer3, scale_factor=4, mode='bilinear')
            nt_layer4_s = F.interpolate(nt_layer4, scale_factor=8, mode='bilinear')
            nt_total = nt_layer1 + nt_layer2_s + nt_layer3_s + nt_layer4_s

            ax = plt.subplot2grid((2,5), (0,0))
            ax.imshow(nt_layer1.squeeze().permute(1,2,0).numpy())
            ax.title.set_text('Neural Texture Layer 1 (dim1-3)')
            ax1 = plt.subplot2grid((2,5), (0,1))
            ax1.imshow(nt_layer2.squeeze().permute(1,2,0).numpy())
            ax1.title.set_text('Neural Texture Layer 2 (dim1-3)')
            ax2 = plt.subplot2grid((2,5), (0,2))
            ax2.imshow(nt_layer3.squeeze().permute(1,2,0).numpy())
            ax2.title.set_text('Neural Texture Layer 3 (dim1-3)')
            ax3 = plt.subplot2grid((2,5), (0,3))
            ax3.imshow(nt_layer4.squeeze().permute(1,2,0).numpy())
            ax3.title.set_text('Neural Texture Layer 4 (dim1-3)')
            ax4 = plt.subplot2grid((2,5), (0,4))
            ax4.imshow(nt_total.squeeze().permute(1,2,0).numpy())
            ax4.title.set_text('Combined Neural Texture')

            n_nt_layer1 = nt_layer1 * 1 
            n_nt_layer2 = nt_layer2 * 1 
            n_nt_layer3 = nt_layer3 * 1 
            n_nt_layer4 = nt_layer4 * 1
            n_nt_total = nt_total * 1

            ax5 = plt.subplot2grid((2,5), (1,0))
            ax5.imshow(n_nt_layer1.squeeze().permute(1,2,0).numpy() * -1)
            ax5.title.set_text('Negative Layer 1 (dim1-3)')
            ax6 = plt.subplot2grid((2,5), (1,1))
            ax6.imshow(n_nt_layer2.squeeze().permute(1,2,0).numpy() * -1)
            ax6.title.set_text('Negative Layer 2 (dim1-3)')
            ax7 = plt.subplot2grid((2,5), (1,2))
            ax7.imshow(n_nt_layer3.squeeze().permute(1,2,0).numpy() * -1)
            ax7.title.set_text('Negative Layer 3 (dim1-3)')
            ax8 = plt.subplot2grid((2,5), (1,3))
            ax8.imshow(n_nt_layer4.squeeze().permute(1,2,0).numpy() * -1)
            ax8.title.set_text('Negative Layer 4 (dim1-3)')
            ax9 = plt.subplot2grid((2,5), (1,4))
            ax9.imshow(n_nt_total.squeeze().permute(1,2,0).numpy() * -1)
            ax9.title.set_text('Combined Negative Texture')

            if args.use_tensorboard:
                self.writer.add_figure('Neural Texture (Dim 13-15)', fig, step, close=True)
            else:
                fig.canvas.draw()
                wandb_plot_b = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                wandb_plot_b = wandb_plot_b.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                wandb.log({"Neural Texture (Dim 13-15)": wandb.Image(wandb_plot_b)})

        # Visualize the inputs and outputs of the neural renderer
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(241)
        ax.imshow(util.denormalize_img(images[0].permute(1,2,0), mean=mean, std=std).numpy())
        ax.title.set_text('Input Image')
        ax2 = fig.add_subplot(242)
        ax2.imshow((util.denormalize_img(RGB_texture[0].permute(1,2,0), mean=mean, std=std)*img_mask[0].permute(1,2,0)).numpy())
        ax2.title.set_text('NT Image w/ Mask')

        ax8 = fig.add_subplot(243)
        ax8.imshow((util.denormalize_img(images[0].permute(1,2,0), mean=mean, std=std)*img_mask[0].permute(1,2,0)).numpy())
        ax8.title.set_text('GT Image w/ Mask')

        ax6 = fig.add_subplot(244)
        ax6.imshow(img_mask[0].squeeze(0).numpy(), cmap=plt.cm.gray)
        ax6.title.set_text('Mask Image')

        ax5 = fig.add_subplot(245)
        ax5.imshow(util.denormalize_img(background[0].permute(1,2,0), mean=mean, std=std).numpy())
        ax5.title.set_text('Background')
        ax7 = fig.add_subplot(246)
        ax7.imshow(util.denormalize_img(preds[0].permute(1,2,0), mean=mean, std=std).numpy())
        ax7.title.set_text('Rendered Image')

        if args.predict_mask:
            ax4 = fig.add_subplot(247)
            temp = torch.clip(util.denormalize_img(preds_rgb[0].detach().cpu().permute(1,2,0), mean=mean, std=std), min=0, max=1).numpy()
            ax4.imshow(util.denormalize_img(preds_rgb[0].detach().cpu().permute(1,2,0), mean=mean, std=std).numpy())
            ax4.title.set_text('Predicted RGB')
            ax10 = fig.add_subplot(248)
            ax10.imshow(preds_mask[0].detach().cpu().squeeze(0).numpy(), cmap=plt.cm.gray)
            ax10.title.set_text("Predicted Mask")
        else:
            ax4 = fig.add_subplot(247)
            ax4.imshow(util.denormalize_img(preds[0].permute(1,2,0), mean=mean, std=std).numpy() * -1)
            ax4.title.set_text('Negative Rendered Image')

        if args.use_tensorboard:
            self.writer.add_figure("Training/Outputs", fig, step, close=True)
        else:
            fig.canvas.draw()
            wandb_plot_c = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            wandb_plot_c = wandb_plot_c.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            wandb.log({"Training/Outputs": wandb.Image(wandb_plot_c)})
    

    def log_validation_losses_loop(self, args, validation_L1_1, validation_L1_2,
                              validation_PSNR_1, validation_PSNR_2,
                              validation_SSIM_1, validation_SSIM_2,
                              epoch):
        if args.use_tensorboard:
            self.writer.add_scalar("Validation Photometric (L1) (t=1)", validation_L1_1.item(), epoch)
            self.writer.add_scalar("Validation PSNR (t=1)", validation_PSNR_1.item(), epoch)
            self.writer.add_scalar("Validation SSIM (t=1)", validation_SSIM_1.item(), epoch)

            self.writer.add_scalar("Validation Photometric (L1) (t=2)", validation_L1_2.item(), epoch)
            self.writer.add_scalar("Validation PSNR (t=2)", validation_PSNR_2.item(), epoch)
            self.writer.add_scalar("Validation SSIM (t=2)", validation_SSIM_2.item(), epoch)
        else:
            wandb.log({"Validation Photometric (L1) (t=1)": validation_L1_1})
            wandb.log({"Validation PSNR (t=1)": validation_PSNR_1})
            wandb.log({"Validation SSIM (t=1)": validation_SSIM_1})

            wandb.log({"Validation Photometric (L1) (t=2)": validation_L1_2})
            wandb.log({"Validation PSNR (t=2)": validation_PSNR_2})
            wandb.log({"Validation SSIM (t=2)": validation_SSIM_2})
        

    def log_validation_images(self, args, images, pred_1, pred_2, pred_1_rgb, pred_1_mask, pred_2_rgb, pred_2_mask, mean, std, epoch):
        """ WEIGHTS AND BIAS LOGGING """
        images = images.detach().cpu()
        pred_1 = pred_1.detach().cpu()
        pred_2 = pred_2.detach().cpu()
        if args.predict_mask:
            pred_1_rgb = pred_1_rgb.detach().cpu()
            pred_1_mask = pred_1_mask.detach().cpu()
            pred_2_rgb = pred_2_rgb.detach().cpu()
            pred_2_mask = pred_2_mask.detach().cpu()

        if args.predict_mask:
            # t=1
            display_images = torch.clip(util.denormalize_img(pred_1_rgb.cpu().permute(0,2,3,1),mean=mean, std=std), min=0, max=1).permute(0,3,1,2)
            display_images = make_grid(display_images, nrow=display_images.shape[0]//2, normalize=False)
            clip_preds = torch.clip(pred_1_mask.cpu().permute(0,2,3,1),min=0, max=1).permute(0,3,1,2)
            clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2, normalize=False,)

            if args.use_tensorboard:
                self.writer.add_figure('Validation/Mask Prediction (t=1)', util.show(clip_preds), epoch, close=True)
                self.writer.add_figure("Validation/RGB Prediction (t=1)", util.show(display_images), epoch, close=True)
            else:
                wandb_image = wandb.Image(clip_preds, caption="Mask Prediction (t=1)") 
                wandb.log({"Validation/Mask Prediction (t=1)": wandb_image}) 
                wandb_image = wandb.Image(display_images, caption="RGB Prediction (t=1)")          
                wandb.log({"Validation/RGB Prediction (t=1)": wandb_image})

            # t=2
            display_images = torch.clip(util.denormalize_img(pred_2_rgb.cpu().permute(0,2,3,1),mean=mean, std=std), min=0, max=1).permute(0,3,1,2)
            display_images = make_grid(display_images, nrow=display_images.shape[0]//2, normalize=False)
            clip_preds = torch.clip(pred_2_mask.cpu().permute(0,2,3,1),min=0, max=1).permute(0,3,1,2)
            clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2, normalize=False,)

            if args.use_tensorboard:
                self.writer.add_figure('Validation/Mask Prediction (t=2)', util.show(clip_preds), epoch, close=True)
                self.writer.add_figure("Validation/RGB Prediction (t=2)", util.show(display_images), epoch, close=True)
            else:
                wandb_image = wandb.Image(clip_preds, caption="Mask Prediction (t=2)")          
                wandb.log({"Validation/Mask Prediction (t=2)": wandb_image})
                wandb_image = wandb.Image(display_images, caption="RGB Prediction (t=2)")          
                wandb.log({"Validation/RGB Prediction (t=2)": wandb_image})

        # visual batch training results (t=1)
        display_images = util.denormalize_img(images[:,0,...].cpu().permute(0,2,3,1),mean=mean, std=std).permute(0,3,1,2)
        difference_display = display_images.clone()
        display_images = make_grid(display_images, nrow=display_images.shape[0]//2)

        clip_preds = torch.clip(util.denormalize_img(pred_1.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1).permute(0,3,1,2)
        difference_clip_preds = clip_preds.clone()
        clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2)

        if args.use_tensorboard:
            self.writer.add_figure("Validation/GT Images (t=1)", util.show(display_images), epoch, close=True)
            self.writer.add_figure('Validation/Predictions-clip_values (t=1)', util.show(clip_preds), epoch, close=True)
        else:
            wandb_image = wandb.Image(display_images, caption="GT Images (t=1)")          
            wandb.log({"Validation/GT (t=1)": wandb_image})
            wandb_image = wandb.Image(clip_preds, caption="Predictions (clip preds + train norm)")          
            wandb.log({"Validation/Predictions-clip_values (t=1)": wandb_image})

        # t=2
        display_images = util.denormalize_img(images[:,1,...].cpu().permute(0,2,3,1),mean=mean, std=std).permute(0,3,1,2)
        difference_display_2 = torch.clone(display_images)
        display_images = make_grid(display_images, nrow=display_images.shape[0]//2)

        clip_preds = torch.clip(util.denormalize_img(pred_2.cpu().permute(0,2,3,1),mean=mean, std=std),min=0, max=1).permute(0,3,1,2)
        difference_clip_preds_2 = clip_preds.clone()
        clip_preds = make_grid(clip_preds, nrow=clip_preds.shape[0]//2)

        if args.use_tensorboard:
            self.writer.add_figure("Validation/GT Images (t=2)", util.show(display_images), epoch, close=True)
            self.writer.add_figure('Validation/Predictions-clip_values (t=2)', util.show(clip_preds), epoch, close=True)
        else:
            wandb_image = wandb.Image(display_images, caption="GT Images (t=2)")          
            wandb.log({"Validation/GT (t=2)": wandb_image})
            wandb_image = wandb.Image(clip_preds, caption="Predictions (clip preds + train norm)")          
            wandb.log({"Validation/Predictions-clip_values (t=2)": wandb_image})

        ddd = make_grid(difference_display - difference_display_2, nrow=difference_display_2.shape[0]//2)
        wandb_image = wandb.Image(ddd, caption="GT Difference Map")  
        if args.use_tensorboard:
            self.writer.add_figure("Validation/GT Difference Map", util.show((ddd-torch.min(ddd))/torch.max(ddd)), epoch, close=True)
        else:        
            wandb.log({"Validation/GT Difference Map": wandb_image})

        dcp = make_grid(difference_clip_preds - difference_clip_preds_2, nrow=difference_clip_preds_2.shape[0]//2)
        wandb_image = wandb.Image(dcp, caption="Pred Difference Map")   
        if args.use_tensorboard:
            self.writer.add_figure("Training/Pred Difference Map", util.show((dcp-torch.min(dcp))/torch.max(dcp)), epoch, close=True)  
        else:          
            wandb.log({"Validation/Pred Difference Map": wandb_image})

    