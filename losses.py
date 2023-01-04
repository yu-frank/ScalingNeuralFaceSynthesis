import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.metrics import structural_similarity

def singlePSNR(y, y_pred):
    # y and y_pred should be denormalized and should be between 0 and 1
    MSE = torch.nn.MSELoss()
    mse = MSE(y, y_pred)
    psnr = 10 * torch.log10(1/mse)
    return psnr

def PSNR(y, y_pred):
    sum = 0
    for i in range(y.shape[0]):
        sum += singlePSNR(y[i], y_pred[i])
    return sum / y.shape[0]

def SSIM(y, y_pred):
    s = 0
    for i in range(y.shape[0]):
        s += structural_similarity(y[i].numpy(), y_pred[i].numpy(), multichannel=True) 
    return s / y.shape[0]