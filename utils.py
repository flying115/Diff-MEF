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


import os
import cv2
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils import data

class load_data_test(data.Dataset):
    def __init__(self, input_data_low, input_data_high, text_file):
        self.input_data_low = input_data_low
        self.input_data_high = input_data_high
        self.text_file = text_file
        print("Total testing examples:", len(self.input_data_high))
        self.text_dict = self.load_text_file(self.text_file)
        self.transform=A.Compose([ToTensorV2(),])

    def __len__(self):
        return len(self.input_data_low)

    def load_text_file(self, text_file):
        text_dict = {}
        with open(text_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(":")
                idx = int(parts[0].strip())
                text = parts[1].strip()
                text_dict[idx] = text
        return text_dict

    def __getitem__(self, idx):
        seed = torch.random.seed()

        data_low = cv2.imread(self.input_data_low[idx])
        data_low = data_low[:,:,::-1].copy()
        random.seed(1)
        data_low = data_low/255.0
        
        h, w, _ = data_low.shape
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        top = (h - new_h) // 2
        bottom = h - new_h - top
        left = (w - new_w) // 2
        right = w - new_w - left
        data_low = data_low[top:h - bottom, left:w - right]
        
        data_low=np.power(data_low,0.5)
        data_low = self.transform(image=data_low)["image"]
        data_low = data_low * 2 - 1

        data_high = cv2.imread(self.input_data_high[idx])
        data_high = data_high[top:h - bottom, left:w - right]
        data_high=data_high[:,:,::-1].copy()
        #data_high = Image.fromarray(data_high)
        random.seed(1)
        data_high = self.transform(image=data_high)["image"]/255.0
        data_high=data_high*2-1
        
        img_idx = int(self.input_data_low[idx].split('/')[-1].split('.')[0])  
        text_description = self.text_dict.get(img_idx, "No description available")
        
        return data_low, data_high, self.input_data_low[idx], text_description


def fast_show_mask_gpu(image, annotation, ax, random_color=False):
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    # Find the index of the first non-zero value at each position.
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color:
        color = torch.rand((mask_sum, 1, 1, 3)).to(annotation.device)
    else:
        color = torch.ones((mask_sum, 1, 1, 3)).to(annotation.device) * torch.tensor([
            30 / 255, 144 / 255, 255 / 255]).to(annotation.device)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(annotation.device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    # Select data according to the index. The index indicates which batch's data to choose at each position, converting the mask_image into a single batch form.
    show = torch.zeros((height, weight, 4)).to(annotation.device)
    try:
        h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
    except:
        h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]
    rgb_image = show[:, :, :3]
    alpha = show[:, :, 3:4]
    rgb_image_weighted = rgb_image * alpha
    rgb_image_weighted = (rgb_image_weighted * 255).clamp(0, 255).to(torch.uint8)
    show_cpu = rgb_image_weighted.cpu().numpy()
    return show_cpu

def plot_to_result(image, annotations, mask_random_color=True, withContours=True) -> np.ndarray:
    if isinstance(image, str) or isinstance(image, Image.Image):
        image = image_to_np_ndarray(image)
    if isinstance(annotations[0], dict):
        annotations = [annotation['segmentation'] for annotation in annotations]
    # original_h = image.shape[0]
    # original_w = image.shape[1]
    # if sys.platform == "darwin":
    #     plt.switch_backend("TkAgg")
    # plt.figure(figsize=(original_w / 100, original_h / 100))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if isinstance(annotations[0], np.ndarray):
        annotations = torch.from_numpy(annotations)
    img_array = fast_show_mask_gpu(image, annotations,plt.gca(), random_color=mask_random_color,)
    # if isinstance(annotations, torch.Tensor):
    #     annotations = annotations.cpu().numpy()
    # plt.axis('off')
    # fig = plt.gcf()
    # plt.draw()
    # try:
    #     buf = fig.canvas.tostring_rgb()
    # except AttributeError:
    #     fig.canvas.draw()
    #     buf = fig.canvas.tostring_rgb()
    # cols, rows = fig.canvas.get_width_height()
    # img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def resize_segmentation_to_input(segmentation, input_image):
    h, w = input_image.shape[-2:]
    resized_seg = cv2.resize(segmentation, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized_seg

@contextlib.contextmanager
def hide_stdout():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            yield

def getSnrMap(data_low,data_blur):
    data_low = data_low[:, 0:1, :, :] * 0.299 + data_low[:, 1:2, :, :] * 0.587 + data_low[:, 2:3, :, :] * 0.114
    data_blur = data_blur[:, 0:1, :, :] * 0.299 + data_blur[:, 1:2, :, :] * 0.587 + data_blur[:, 2:3, :, :] * 0.114
    noise = torch.abs(data_low - data_blur)

    mask = torch.div(data_blur, noise + 0.0001)

    batch_size = mask.shape[0]
    height = mask.shape[2]
    width = mask.shape[3]
    mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
    mask_max = mask_max.view(batch_size, 1, 1, 1)
    mask_max = mask_max.repeat(1, 1, height, width)
    mask = mask * 1.0 / (mask_max + 0.0001)

    mask = torch.clamp(mask, min=0, max=1.0)
    mask = mask.float()
    return mask

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_color_map(im):
    return im / (rgb2gray(im)[..., np.newaxis] + 1e-6) * 100
    # return im / (np.mean(im, axis=-1)[..., np.newaxis] + 1e-6) * 100


def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calculate_ssim(img1, img2):
    score, _ = SSIM(img1, img2, full=True)
    return score