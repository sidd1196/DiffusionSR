import torch 
import torch.nn as nn
import os
import random
import math
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from config import (
    patch_size, scale, dir_HR, dir_LR, dir_valid_HR, dir_valid_LR, 
    _project_root, device, gt_size
)
from realesrgan import RealESRGANDegrader
from autoencoder import get_vqgan

# Initialize degradation pipeline and VQGAN (lazy loading)
_degrader = None
_vqgan = None

def get_degrader():
    """Get or create degradation pipeline."""
    global _degrader
    if _degrader is None:
        _degrader = RealESRGANDegrader(scale=scale)
    return _degrader

def get_vqgan_model():
    """Get or create VQGAN model."""
    global _vqgan
    if _vqgan is None:
        _vqgan = get_vqgan(device=device)
    return _vqgan


class SRDatasetOnTheFly(torch.utils.data.Dataset):
    """
    PyTorch Dataset for on-the-fly degradation and VQGAN encoding.
    
    This dataset:
    1. Loads full HR images
    2. Crops 256x256 patches on-the-fly
    3. Applies RealESRGAN degradation to generate LR
    4. Upsamples LR to 256x256 using bicubic
    5. Encodes both HR and LR through VQGAN to get 64x64 latents
    
    Args:
        dir_HR (str): Directory path containing high-resolution images.
        scale (int, optional): Super-resolution scale factor. Defaults to config.scale (4).
        patch_size (int, optional): Size of patches. Defaults to config.patch_size (256).
        max_samples (int, optional): Maximum number of images to load. If None, loads all.
    
    Returns:
        tuple: (hr_latent, lr_latent) where both are torch.Tensor of shape (C, 64, 64)
               representing VQGAN-encoded latents.
    """
    
    def __init__(self, dir_HR, scale=scale, patch_size=patch_size, max_samples=None):
        super().__init__()
        
        self.dir_HR = dir_HR
        self.scale = scale
        self.patch_size = patch_size
        
        # Get all image files
        self.filenames = sorted([
            f for f in os.listdir(self.dir_HR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Limit to max_samples if specified
        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]
        
        # Initialize degradation and VQGAN (will be loaded on first use)
        self.degrader = None
        self.vqgan = None
    
    def __len__(self):
        return len(self.filenames)
    
    def _load_image(self, img_path):
        """Load and validate image."""
        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img)  # (C, H, W) in range [0, 1]
        return img_tensor
    
    def _crop_patch(self, img_tensor, patch_size):
        """
        Crop a random patch from image.
        
        Args:
            img_tensor: (C, H, W) tensor
            patch_size: Size of patch to crop
        
        Returns:
            patch: (C, patch_size, patch_size) tensor
        """
        C, H, W = img_tensor.shape
        
        # Pad if image is smaller than patch_size
        if H < patch_size or W < patch_size:
            pad_h = max(0, patch_size - H)
            pad_w = max(0, patch_size - W)
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = img_tensor.shape[1], img_tensor.shape[2]
        
        # Random crop
        top = random.randint(0, max(0, H - patch_size))
        left = random.randint(0, max(0, W - patch_size))
        
        patch = img_tensor[:, top:top+patch_size, left:left+patch_size]
        return patch
    
    def _apply_augmentations(self, hr, lr):
        """
        Apply synchronized augmentations to HR and LR.
        
        Args:
            hr: (C, H, W) HR tensor
            lr: (C, H, W) LR tensor
        
        Returns:
            hr_aug, lr_aug: Augmented tensors
        """
        # Horizontal flip
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[2])
            lr = torch.flip(lr, dims=[2])
        
        # Vertical flip
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[1])
            lr = torch.flip(lr, dims=[1])
        
        # 180° rotation
        if random.random() < 0.5:
            hr = torch.rot90(hr, k=2, dims=[1, 2])
            lr = torch.rot90(lr, k=2, dims=[1, 2])
        
        return hr, lr
    
    def __getitem__(self, idx):
        # Load HR image
        hr_path = os.path.join(self.dir_HR, self.filenames[idx])
        hr_full = self._load_image(hr_path)  # (C, H, W) in [0, 1]
        
        # Crop 256x256 patch from HR
        hr_patch = self._crop_patch(hr_full, self.patch_size)  # (C, 256, 256)
        
        # Initialize degrader and VQGAN on first use
        if self.degrader is None:
            self.degrader = get_degrader()
        if self.vqgan is None:
            self.vqgan = get_vqgan_model()
        
        # Apply degradation on-the-fly to generate LR
        # Degrader expects (C, H, W) and returns (C, H//scale, W//scale)
        hr_patch_gpu = hr_patch.to(device)  # (C, 256, 256)
        with torch.no_grad():
            lr_patch = self.degrader.degrade(hr_patch_gpu)  # (C, 64, 64) in pixel space
        
        # Upsample LR to 256x256 using bicubic interpolation
        lr_patch_upsampled = F.interpolate(
            lr_patch.unsqueeze(0),  # (1, C, 64, 64)
            size=(self.patch_size, self.patch_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)  # (C, 256, 256)
        
        # Apply augmentations (synchronized)
        hr_patch, lr_patch_upsampled = self._apply_augmentations(
            hr_patch.cpu(), 
            lr_patch_upsampled.cpu()
        )
        
        # Encode through VQGAN to get latents (64x64)
        # Move to device for encoding
        hr_patch_gpu = hr_patch.to(device).unsqueeze(0)  # (1, C, 256, 256)
        lr_patch_gpu = lr_patch_upsampled.to(device).unsqueeze(0)  # (1, C, 256, 256)
        
        with torch.no_grad():
            # Encode HR: 256x256 -> 64x64 latent
            hr_latent = self.vqgan.encode(hr_patch_gpu)  # (1, C, 64, 64)
            
            # Encode LR: 256x256 -> 64x64 latent
            lr_latent = self.vqgan.encode(lr_patch_gpu)  # (1, C, 64, 64)
        
        # Remove batch dimension and move to CPU
        hr_latent = hr_latent.squeeze(0).cpu()  # (C, 64, 64)
        lr_latent = lr_latent.squeeze(0).cpu()  # (C, 64, 64)
        
        return hr_latent, lr_latent


# Create datasets using on-the-fly processing
train_dataset = SRDatasetOnTheFly(
    dir_HR=dir_HR,
    scale=scale,
    patch_size=patch_size
)

valid_dataset = SRDatasetOnTheFly(
    dir_HR=dir_valid_HR,
    scale=scale,
    patch_size=patch_size
)

# Mini dataset with 8 images for testing
mini_dataset = SRDatasetOnTheFly(
    dir_HR=dir_HR,
    scale=scale,
    patch_size=patch_size,
    max_samples=8
)

print(f"\nFull training dataset size: {len(train_dataset)}")
print(f"Full validation dataset size: {len(valid_dataset)}")
print(f"Mini dataset size: {len(mini_dataset)}")
print(f"Using on-the-fly degradation and VQGAN encoding")
print(f"Output: 64x64 latents (from 256x256 patches)")
