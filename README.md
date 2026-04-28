# Diff-MEF
Diff-MEF: Cross-Modal Diffusion Framework With Text Prompts and Semantic Perception for Multi-Exposure Image Fusion（TIP2026）
---

## News

- The inference code and environment configuration have been released.
- Pretrained checkpoints are provided with Git LFS.
- More training details and benchmark results will be updated later.

---

## Overview

Multi-exposure image fusion aims to generate a well-exposed image from multiple images captured under different exposure conditions. Diff-MEF uses a cross-modal diffusion framework with:

- text prompt guidance;
- diffusion-based image generation;
- low-exposure and over-exposure image fusion;
- segmentation-guided feature refinement.

The current repository supports quick testing with pretrained checkpoints.

---

## Repository Structure

```text
Diff-MEF/
├── Diffusion/                  # Diffusion model, UNet, TDRM and encoders
├── fastsam/                    # FastSAM-related modules
├── loss/                       # Loss functions
├── ckpt/                       # Pretrained checkpoints
│   ├── ckpt_unet.pt
│   └── TDRM/
│       └── ckpt_tdrm.pt
├── dataset/
│   └── test/
│       └── SICE/
│           ├── ue/             # Under-exposed images
│           ├── oe/             # Over-exposed images
│           └── SICE_text.txt   # Text prompts
│       └── MEFB/
│           ├── ue/             # Under-exposed images
│           ├── oe/             # Over-exposed images
│           └── MEFB_text.txt   # Text prompts
├── results/                    # Inference results
├── environment.txt             # Environment package versions
├── test.py                     # Inference script
├── utils.py
└── README.md
