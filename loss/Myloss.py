import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from kornia.losses import ssim_loss

def grad_loss(image_fuse, gt_image):
    sobel_x = torch.FloatTensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(gt_image.device)
    sobel_y = torch.FloatTensor([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(gt_image.device)
    padding = (1, 1, 1, 1)

    def gradient(image):
        image = F.pad(image, padding, mode='replicate')
        gradient_x = F.conv2d(image, sobel_x, padding=0, groups=3)
        gradient_y = F.conv2d(image, sobel_y, padding=0, groups=3)
        return torch.abs(gradient_x), torch.abs(gradient_y)

    gradient_gt_x, gradient_gt_y = gradient(gt_image)
    gradient_fuse_x, gradient_fuse_y = gradient(image_fuse)
    loss = F.l1_loss(gradient_fuse_x + gradient_fuse_y, gradient_gt_x + gradient_gt_y)
    return loss

def contrastive_loss(img_feas, text_feas, tau=0.2):
    img_feas = F.normalize(img_feas, dim=1)  # shape (16, 512)
    text_feas = F.normalize(text_feas, dim=1)

    positive_text_fea = text_feas[0,:].unsqueeze(0)
    negative_text_feas = text_feas[1:, :]

    positive_sim = torch.matmul(img_feas, positive_text_fea.T)/tau
    negative_sim = torch.matmul(img_feas, negative_text_feas.T)/tau

    # similarities = torch.cat([positive_sim, negative_sim], dim=1)
    # labels = torch.zeros(img_feas.size(0), dtype=torch.long).to(img_feas.device)
    # contrast_loss = torch.mean(F.cross_entropy(similarities, labels))

    for si in range(img_feas.size(0)):
        pos = positive_sim[si, 0:1]
        negs = negative_sim[si, :]
        softmax = - torch.logsumexp(pos, 0) + torch.logsumexp(negs, 0)
        softmax = softmax.unsqueeze(-1)
        if si == 0:
            softmaxes = softmax
        else:
            softmaxes = torch.cat((softmaxes, softmax), 0)

    contrast_loss = torch.mean(softmaxes)
    return contrast_loss



def RGB2YCrCb(rgb_image):

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    # Clamping the values to [0, 1]
    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()  # detach to avoid gradient flow
    Cb = Cb.clamp(0.0, 1.0).detach() # detach to avoid gradient flow
    
    return Y, Cb, Cr


def color_loss(image_fuse, gt_image):
    Y_fuse, Cb_fuse, Cr_fuse = RGB2YCrCb(image_fuse)
    Y_gt, Cb_gt, Cr_gt = RGB2YCrCb(gt_image)

    loss_func = nn.L1Loss(reduction='mean').to(gt_image.device)
    loss_color = (loss_func(Cb_fuse, Cb_gt) + loss_func(Cr_fuse, Cr_gt))
    loss_color = loss_color * 50#25
    return loss_color

def exp_loss(image_fuse, gt_image):
    Y_fuse, Cb_fuse, Cr_fuse = RGB2YCrCb(image_fuse)
    Y_gt, Cb_gt, Cr_gt = RGB2YCrCb(gt_image)

    loss_func = nn.L1Loss(reduction='mean').to(gt_image.device)
    loss_exp = loss_func(Y_fuse, Y_gt)
    loss_exp = loss_exp * 10
    return loss_exp
