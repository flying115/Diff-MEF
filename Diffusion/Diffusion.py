from typing import Dict
from tensorboardX import SummaryWriter
import cv2
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# from matplotlib.animation import FuncAnimation
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from loss import Myloss
from kornia.losses import ssim_loss
import numpy as np
import lpips
import clip
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


def preprocess_tensor(tensor):
    resize = transforms.Resize((224, 224))
    tensor = resize(tensor)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    tensor = normalize(tensor)
    return tensor


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)  
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))  



class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, Unet_model,Refine_model, image_encoder, seg_encoder, beta_1, beta_T, T, clip_model, text_features,Pre_train=None):
        super().__init__()

        self.unet = Unet_model
        self.refine = Refine_model
        self.clip_model = clip_model
        self.image_encoder = image_encoder
        self.seg_encoder = seg_encoder
        
        self.text_features = text_features
        
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # self.alphas_bar=alphas_bar
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.num = 0

    def initialize_model_weights(self, init_type='orthogonal'):
        #init_weights(self.unet, init_type=init_type)
        init_weights(self.refine, init_type=init_type)
        if hasattr(self, 'image_encoder'):
            init_weights(self.image_encoder, init_type=init_type)
        if hasattr(self, 'seg_encoder'):
            init_weights(self.seg_encoder, init_type=init_type)
        
    def forward(self, ue_image, oe_image, gt_image, text_emb, ue_seg, oe_seg, epoch):
        ue_image = ue_image.float()
        oe_image = oe_image.float()
        gt_image = gt_image.float()
        ue_seg = ue_seg.float()
        oe_seg = oe_seg.float()
        text_emb = text_emb.float()
        
        t = torch.randint(self.T, size=(gt_image.shape[0],), device=gt_image.device)
        noise = torch.randn_like(gt_image)
        y_t = (extract(self.sqrt_alphas_bar, t, gt_image.shape) * gt_image +
               extract(self.sqrt_one_minus_alphas_bar, t, gt_image.shape) * noise)
               
        input = torch.cat([ue_image, oe_image, y_t], dim=1)

        noise_pred = self.unet(input, t)

        y_0_pred = 1 / extract(self.sqrt_alphas_bar, t, gt_image.shape) * (
                y_t - extract(self.sqrt_one_minus_alphas_bar, t, gt_image.shape) * noise_pred).float()
                
        seg_features = self.seg_encoder(torch.cat([ue_seg, oe_seg], dim=1))
        img_features = self.image_encoder(torch.cat([ue_image,oe_image], dim=1))
        
        data_concate = torch.cat([img_features, seg_features, ue_image, oe_image], dim=1)
        
        y_0_refine = self.refine(y_0_pred.detach(), data_concate, t, text_emb)

        loss_mse = torch.mean(F.mse_loss(noise_pred, noise, reduction='none'))
        
        loss_denoise = 8 * loss_mse + 2 * F.l1_loss(y_0_pred, gt_image)
         
        loss_l1 =  F.l1_loss(y_0_refine, gt_image)
        
        loss_ssim = torch.mean(ssim_loss(y_0_refine, gt_image, window_size=11))
        
        loss_grad = Myloss.grad_loss(y_0_refine, gt_image)
        
        loss_color = Myloss.color_loss(y_0_refine, gt_image)

        with torch.no_grad():
            y_0_refine_features = self.clip_model.encode_image(y_0_refine)
        loss_text = Myloss.contrastive_loss(y_0_refine_features, self.text_features)

        loss = loss_denoise + 4 * loss_ssim  +  5 * loss_l1 + 10 * loss_grad + loss_color + 0.001 * loss_text
        
        return [loss, loss_denoise, loss_ssim, loss_l1, loss_grad, loss_color, loss_text, y_0_pred, y_0_refine, img_features, seg_features]



class GaussianDiffusionSampler(nn.Module):
    def __init__(self, Unet_model, Refine_model, image_encoder, seg_encoder, beta_1, beta_T, T):
        super().__init__()

        self.unet = Unet_model
        self.refine = Refine_model
        self.image_encoder = image_encoder
        self.seg_encoder = seg_encoder
        
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.sqrt_alphas_bar = alphas_bar
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
        self.alphas_bar = alphas_bar
        self.one_minus_alphas_bar = (1. - alphas_bar)
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.grad_coeff = None
        self.every_mins = 0

    def predict_xt_prev_mean_from_eps(self, t, eps, y_t):
        assert y_t.shape == eps.shape
        return (
                extract(self.coeff1, t, y_t.shape) * y_t -
                extract(self.coeff2, t, y_t.shape) * eps
        )

    def p_mean_variance(self, input, t, y_t):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, input.shape)
        eps = self.unet(input, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(t, eps, y_t)

        return xt_prev_mean, var

    def forward(self, ue_image, oe_image, text_emb,ue_seg, oe_seg,  ddim=False, unconditional_guidance_scale=1, ddim_step=None):
        ue_image = ue_image.float()
        oe_image = oe_image.float()
        ue_seg = ue_seg.float()
        oe_seg = oe_seg.float()
        text_emb = text_emb.float()
        
        if ddim == False:
            device = ue_image.device
            noise = torch.randn_like(ue_image).to(device)
            y_t = noise
            for time_step in reversed(range(self.T)):
                t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step
                
        
                input = torch.cat([ue_image,oe_image, y_t], dim=1).float() 
                mean, var = self.p_mean_variance(input, t, y_t)
                if time_step > 0:
                    noise = torch.randn_like(y_t)
                else:
                    noise = 0
                y_t = mean + torch.sqrt(var) * noise
                # assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."

            y_0 = y_t
            return torch.clip(y_0, -1, 1)

        else:
            device = ue_image.device
            noise = torch.randn_like(ue_image).to(device)
            y_t = noise

            step = 1000 / ddim_step
            step = int(step)
            seq = range(0, 1000, step)
            seq_next = [-1] + list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(y_t.shape[0]) * i).to(device).long()
                next_t = (torch.ones(y_t.shape[0]) * j).to(device).long()
                at = extract(self.alphas_bar.to(device), (t + 1).long(), y_t.shape)
                at_next = extract(self.alphas_bar.to(device), (next_t + 1).long(), y_t.shape)
        
                input = torch.cat([ue_image,oe_image, y_t], dim=1).float() 
                eps = self.unet(input, t)

                # classifier free guide
                if unconditional_guidance_scale != 1:
                    eps_unconditional = self.unet(input, t)
                    eps = eps_unconditional + unconditional_guidance_scale * (eps - eps_unconditional)

                y0_pred = (y_t - eps * (1 - at).sqrt()) / at.sqrt()
                
                seg_features = self.seg_encoder(torch.cat([ue_seg, oe_seg], dim=1))
                img_features = self.image_encoder(torch.cat([ue_image,oe_image], dim=1))
        
                data_concate = torch.cat([img_features, seg_features, ue_image, oe_image], dim=1)

                y_0_refine = self.refine( y0_pred, data_concate, t,text_emb)
                
                eta = 0
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                y_t = at_next.sqrt() * y_0_refine + c1 * torch.randn_like(ue_image) + c2 * eps
            y_0 = y_t
            return torch.clip(y_0, -1, 1), seg_features, img_features
