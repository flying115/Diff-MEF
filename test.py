import colorsys
import os
from typing import Dict, List
import PIL
import lpips as lpips
from PIL import Image
import clip
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import albumentations as A
from Diffusion.Diffusion_text import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Unet import UNet
from Diffusion.TDRM import RefineNet
from Diffusion.Image_encoder import ImageEncoder
from Diffusion.Seg_encoder import SegEncoder
from Scheduler import GradualWarmupScheduler
from loss import Myloss
import numpy as np
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import torch.utils.data as data
import glob
import sys
import random
from albumentations.pytorch import ToTensorV2
import lpips
import time
import os
import re
import argparse
import torchvision
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
from fastsam import FastSAM#, FastSAMPrompt
import contextlib
from itertools import chain
from utils import *

import os
import cv2
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils import data


def Test(config: Dict):
    device = config.device_list[0]

    def get_valid_files(path_dir):
        return [
            f for f in glob.glob(os.path.join(path_dir, '*'))
            if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.JPG'}
        ]

    test_low_dir = os.path.join(config.dataset_path, 'ue')
    test_high_dir = os.path.join(config.dataset_path, 'oe')

    datapath_test_low = get_valid_files(test_low_dir)
    datapath_test_high = get_valid_files(test_high_dir)

    test_text_path = config.dataset_path + r'/SICE_text.txt'
    datapath_test_text = test_text_path

    dataload_test = load_data_test(datapath_test_low, datapath_test_high, datapath_test_text)

    dataloader = DataLoader(dataload_test, batch_size=1, num_workers=1, drop_last=True, pin_memory=True)

    clip_model, _ = clip.load("ViT-B/32", device=device)

    # Set up the model
    Unet_model = UNet(
        T=config.T,
        ch=128,
        channel_mults=[1, 2, 3, 4],
        attn=[2],
        num_res_blocks=2,
        dropout=0.15
    )

    Refine_model = RefineNet(
        in_channel=config.in_channel,
        T=config.T,
        dim=config.dim,
        num_blocks=config.num_blocks,
        heads=config.heads,
        ffn_expansion_factor=config.ffn_expansion_factor,
        padding_mode=config.padding_mode,
        bias=config.bias
    )

    image_encoder = ImageEncoder()
    seg_encoder = SegEncoder()

    ckpt_files = [
        os.path.join(config.training_path, f)
        for f in os.listdir(config.training_path)
        if os.path.isfile(os.path.join(config.training_path, f))
    ]
    ckpt_files.sort(key=lambda x: os.path.getmtime(x))

    sam_model = FastSAM('./fastsam/weights/FastSAM-s.pt')
    sam_model.to(config.device_list[0])

    # Iterate over all checkpoint files
    for ckpt_path in ckpt_files:
        print(f"Loading model from {ckpt_path}")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        Refine_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['RefineNet'].items()})
        Unet_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['Unet'].items()})
        image_encoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['ImageEncoder'].items()})
        seg_encoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['SegEncoder'].items()})
        print(f"Model loaded from {ckpt_path}")

        Refine_model.eval()
        Unet_model.eval()
        image_encoder.eval()
        seg_encoder.eval()

        sampler = GaussianDiffusionSampler(Unet_model, Refine_model, image_encoder, seg_encoder, config.beta_1,
                                           config.beta_T, config.T).to(device)

        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        model_save_dir = os.path.join(config.output_path, 'test-text_results', model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        with torch.no_grad():
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                for data_low, data_high, filename, text_description in tqdmDataLoader:
                    name = filename[0].split('/')[-1]
                    print(f"Testing image: {name}")

                    low_image = data_low.to(device)
                    high_image = data_high.to(device)

                    print(text_description)

                    text = clip.tokenize(text_description).to(device)
                    text_emb = clip_model.encode_text(text)

                    # Process images with SAM model
                    seg_low_batch = np.power((((data_low.permute(0, 2, 3, 1) + 1) / 2).cpu()), 2).numpy() * 255

                    seg_high_batch = ((high_image.permute(0, 2, 3, 1) + 1) / 2).cpu().numpy() * 255

                    seg_low_input = seg_low_batch[0, :, :, :]
                    seg_high_input = seg_high_batch[0, :, :, :]

                    everything_results_low = sam_model(seg_low_input, device=device, retina_masks=False,
                                                       imgsz=data_low.size(2), conf=0.4, iou=0.9)

                    if everything_results_low is None:
                        seg_low = np.zeros((data_low.size(2), data_low.size(3), 3), dtype=np.float64)
                    else:
                        ann_low = everything_results_low[0].masks.data
                        seg_low = plot_to_result(seg_low_input, annotations=ann_low) / 255.0
                        seg_low = resize_segmentation_to_input(seg_low, data_low)

                    everything_results_high = sam_model(seg_high_input, device=device, retina_masks=False,
                                                        imgsz=data_low.size(2), conf=0.4, iou=0.9)

                    if everything_results_high is None:
                        seg_high = np.zeros((data_low.size(2), data_low.size(3), 3), dtype=np.float64)
                    else:
                        ann_high = everything_results_high[0].masks.data
                        seg_high = plot_to_result(seg_high_input, annotations=ann_high) / 255.0
                        seg_high = resize_segmentation_to_input(seg_high, data_high)

                    data_low_seg = torch.tensor(seg_low).permute(2, 0, 1).unsqueeze(0).to(device)
                    data_high_seg = torch.tensor(seg_high).permute(2, 0, 1).unsqueeze(0).to(device)

                    # Inference
                    time_start = time.time()

                    sampledImgs, seg_features, img_features = sampler(
                        low_image,
                        high_image,
                        text_emb,
                        data_low_seg,
                        data_high_seg,
                        ddim=True,
                        unconditional_guidance_scale=1,
                        ddim_step=config.ddim_step
                    )

                    time_end = time.time()
                    print(f"Time cost: {time_end - time_start}")

                    # Post-process
                    sampledImgs = (sampledImgs + 1) / 2
                    low_image = (low_image + 1) / 2
                    high_image = (high_image + 1) / 2

                    res_Imgs = np.clip(
                        sampledImgs.detach().cpu().numpy()[0].transpose(1, 2, 0), 0, 1)[:, :, ::-1]
                    low_img = np.clip(low_image.detach().cpu().numpy()[0].transpose(1, 2, 0), 0, 1)[:, :, ::-1]
                    high_img = np.clip(
                        high_image.detach().cpu().numpy()[0].transpose(1, 2, 0), 0, 1)[:, :, ::-1]

                    # Save results
                    res_Imgs = (res_Imgs * 255)
                    low_img = (low_img * 255)

                    save_path_img = os.path.join(model_save_dir, 'img', name)
                    save_path_seg = os.path.join(model_save_dir, 'seg', name)

                    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
                    os.makedirs(os.path.dirname(save_path_seg), exist_ok=True)

                    save_path = os.path.join(model_save_dir, name)
                    cv2.imwrite(save_path, res_Imgs)

                    img_features_to_save = img_features.cpu().detach().numpy()[0]
                    seg_features_to_save = seg_features.cpu().detach().numpy()[0]

                    img_features_to_save = (img_features_to_save * 255).astype(np.uint8)
                    seg_features_to_save = (seg_features_to_save * 255).astype(np.uint8)

                    img_features_to_save = img_features_to_save.transpose(1, 2, 0)
                    seg_features_to_save = seg_features_to_save.transpose(1, 2, 0)

                    cv2.imwrite(save_path_img.replace('.png', '_features.png'), img_features_to_save)

                    cv2.imwrite(save_path_seg.replace('.png', '_features.png'), seg_features_to_save)

    return None

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    modelConfig = {
  
        "DDP": False,
        "state": "train", # or eval
        "epoch": 1, #10001,
        "batch_size": 1 ,
        "T": 1000,
        "in_channel": 15,
        "dim": 48,
        "num_blocks": [2, 2, 2], #[1, 1, 2, 4], 
        "heads": [1, 2, 4],#[1, 2, 4, 8],
        "ffn_expansion_factor": 2.66,
        "padding_mode": 'reflect',
        "bias": False,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "device_list": [0],
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":2
    }
    
    parser.add_argument('--dataset_path', type=str, default="./dataset/test/SICE/")
    parser.add_argument('--pretrained_path', type=str, default="./ckpt/ckpt_unet.pt")
    parser.add_argument('--training_path', type=str, default="./ckpt/TDRM/")
    parser.add_argument('--output_path', type=str, default="./results/test/")

    config = parser.parse_args()
    
    for key, value in modelConfig.items():
        setattr(config, key, value)
    print(config)
    
    Test(config)