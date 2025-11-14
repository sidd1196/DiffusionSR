import torch 
import torch.nn as nn
import os
import random
from PIL import Image
import torchvision.transforms.functional as TF
from config import patch_size, scale, dir_HR, dir_LR, dir_valid_HR, dir_valid_LR, _project_root

class SRDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading pre-generated HR and LR image patches(from ESRGAN process) for super-resolution.
    
    This dataset expects pre-generated 256x256 patches where:
    - HR patches are high-resolution ground truth images (256x256)
    - LR patches are low-resolution images that have been degraded and upsampled back to 256x256
    - Both HR and LR patches share the same filenames in their respective directories
    
    The dataset applies random augmentations (horizontal flip, vertical flip, 180° rotation)
    synchronously to both HR and LR patches to maintain spatial correspondence.
    
    Args:
        dir_HR (str): Directory path containing high-resolution patches (256x256 PNG files).
        dir_LR (str): Directory path containing low-resolution patches (256x256 PNG files).
        scale (int, optional): Super-resolution scale factor. Defaults to config.scale (4).
        patch_size (int, optional): Size of patches. Defaults to config.patch_size (256).
        max_samples (int, optional): Maximum number of samples to load. If None, loads all patches.
                                    Useful for creating smaller test datasets. Defaults to None.
    
    Returns:
        tuple: (hr_tensor, lr_tensor) where both are torch.Tensor of shape (C, H, W) with
               values in range [0, 1]. C=3 (RGB), H=W=256.
    
    Example:
        >>> dataset = SRDataset(
        ...     dir_HR='/path/to/hr_patches',
        ...     dir_LR='/path/to/lr_patches',
        ...     max_samples=100
        ... )
        >>> hr, lr = dataset[0]  # Get first patch pair
        >>> print(hr.shape, lr.shape)  # torch.Size([3, 256, 256]) torch.Size([3, 256, 256])
    """
    def __init__(self, dir_HR, dir_LR, scale = scale, patch_size = patch_size, max_samples = None):
        super().__init__()

        self.dir_HR = dir_HR
        self.dir_LR = dir_LR
        self.scale = scale
        self.patch_size = patch_size
        
        # For pre-generated patches: HR and LR have same filenames, both are 256x256
        self.filenames = sorted([f for f in os.listdir(self.dir_HR) if f.endswith('.png')])
        
        # Limit to max_samples if specified
        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.dir_HR, self.filenames[idx])
        
        # Pre-generated patches: same filename for both HR and LR (both 256x256)
        lr_path = os.path.join(self.dir_LR, self.filenames[idx])
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        
        # Both are already 256x256 patches, no cropping needed
        # Just apply augmentations

        # ------------------------------
        # Random flip/rotate augmentations
        # ------------------------------
        if random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        if random.random() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)
        if random.random() < 0.5:
            lr = lr.rotate(180)
            hr = hr.rotate(180)

        # ------------------------------
        # Convert to tensors [0,1]
        # ------------------------------
        lr_t = TF.to_tensor(lr)
        hr_t = TF.to_tensor(hr)

        return hr_t, lr_t

train_dataset = SRDataset(
    dir_HR=dir_HR,
    dir_LR=dir_LR,
    scale=scale,
    patch_size=patch_size
)

valid_dataset = SRDataset(
    dir_HR=dir_valid_HR,
    dir_LR=dir_valid_LR,
    scale=scale,
    patch_size=patch_size
)

# Mini dataset with 8 images for testing
mini_dataset = SRDataset(
    dir_HR=dir_HR,
    dir_LR=dir_LR,
    scale=scale,
    patch_size=patch_size,
    max_samples=8
)

print(f"\nFull training dataset size: {len(train_dataset)}")
print(f"\nFull validation dataset size: {len(valid_dataset)}")
print(f"\nMini dataset size: {len(mini_dataset)}")
