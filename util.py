import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import imageio

DATASET_STATS = {'imagenet': "0.485, 0.456, 0.406, 0.229, 0.224, 0.225",}

def remove_t1_data(data) -> dict:
    """
    Modify data from dataloader s.t. it only has t (note will miss last frame) HACK
    """
    for key, value in data.items():
        data[key] = value[:, 0, ...]
    return data

def save_video(save_location, image_list, fps):
    """
    Function to save an output video

    INPUT: 
    save_location (String): where to save the video
    image_list (List): list of np arrays (HxWx3) of type int
    fps (Float): the frame rate of the output video
    """
    imageio.mimwrite(save_location, image_list, "mp4", fps=fps)
    print('Video Saved at {}'.format(save_location))


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = Ft.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

def get_hierarchical_nt_layers(model):
    """given a model it returns the 4 hierarchical nt layers"""
    nt_layer1 = []
    nt_layer2 = []
    nt_layer3 = []
    nt_layer4 = []
    for dim in range(3):
        nt_layer1.append(model.texture.layer1[dim])
        nt_layer2.append(model.texture.layer2[dim])
        nt_layer3.append(model.texture.layer3[dim])
        nt_layer4.append(model.texture.layer4[dim])
    nt_layer1 = torch.stack(nt_layer1).squeeze().unsqueeze(0)
    nt_layer2 = torch.stack(nt_layer2).squeeze().unsqueeze(0)
    nt_layer3 = torch.stack(nt_layer3).squeeze().unsqueeze(0)
    nt_layer4 = torch.stack(nt_layer4).squeeze().unsqueeze(0)
    return nt_layer1, nt_layer2, nt_layer3, nt_layer4

def zero_to_one(tensor):
    """
    convert tensor [-1, 1] to [0, 1]
    """
    return (tensor + 1.) / 2.

def denormalize_img(img_tensor, mean, std, return_tensor_shape=False): 
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)
    if not return_tensor_shape:
        return img_tensor * std + mean
    else:
        return (img_tensor * std + mean).permute(2,0,1)

def img_transform(image, MEAN, STD):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN,
                             std=STD)
    ])
    image = image_transforms(image)
    return image

def map_transform(map):
    map = torch.from_numpy(map)
    return map

def bg_transform(bg, MEAN, STD):
    bg = torch.FloatTensor(bg)
    mean = torch.FloatTensor(MEAN)
    std = torch.FloatTensor(STD)
    return ((bg - mean) / std).permute(2,0,1)

def augment(img, map, crop_size, use_aug, img_mask, mean, std, bg=None, return_aug_crop=False, given_crop=None):
    '''
    :param img:  PIL input image
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''

    if use_aug:
        if given_crop is not None:
            w1, h1, crop_w, crop_h = given_crop
            img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
            if bg is not None:
                bg = bg[h1:h1 + crop_h, w1:w1 + crop_w, :]
            map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
            img_mask = img_mask[h1:h1 + crop_h, w1:w1 + crop_w, :] #already a tensor
        else:
            w, h = img.size
            crop_h, crop_w = crop_size
            w1 = random.randint(0, w - crop_w)

            center_y = random.randint(-crop_h/2,h-crop_h/2)
            h1 = min(max(center_y,0),512-crop_h)            

            img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
            if bg is not None:
                bg = bg[h1:h1 + crop_h, w1:w1 + crop_w, :]
            map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
            img_mask = img_mask[h1:h1 + crop_h, w1:w1 + crop_w, :] #already a tensor

    img, map = img_transform(img, mean, std), map_transform(map)
    if bg is not None:
        bg = bg_transform(bg, mean, std)

    img_mask = map_transform(img_mask)
    img_mask = img_mask.permute(2,0,1)

    if return_aug_crop:
        return img, map, img_mask, bg, w1, h1, crop_w, crop_h
    else:
        return img, map, img_mask, bg

def denormalize_img_cuda(img_tensor, mean, std, return_tensor_shape=False): 
    if not return_tensor_shape:
        return img_tensor * std + mean
    else:
        return (img_tensor * std + mean).permute(2,0,1)