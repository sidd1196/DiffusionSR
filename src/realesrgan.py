import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
# Import Real-ESRGAN's actual degradation pipeline
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_mixed_kernels, random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torch.nn import functional as F_torch

# Apply compatibility patch (if needed)
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except:
    pass

class RealESRGANDegrader:
    """Real-ESRGAN degradation pipeline for a SMOOTH BLUR"""
    
    def __init__(self, scale=4):
        self.scale = scale
        
        # Initialize JPEG compression
        self.jpeger = DiffJPEG(differentiable=False)
        
        # Blur settings (from Real-ESRGAN config)
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        
        # --- Settings for "Smooth Blur" ---
        self.blur_sigma = [2.0, 8.0]  # High blur
        self.noise_range = [1, 5] # Low noise
        self.poisson_scale_range = [0.05, 0.5] # Low noise
        self.jpeg_range = [80, 95] # High quality (low artifact)

        # --- THIS IS THE FIX ---
        # Re-adding the missing attributes
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        # ----------------------
        
        # --- Second degradation settings ---
        self.second_blur_prob = 0.8
        self.blur_kernel_size2 = 21
        self.blur_sigma2 = [1.0, 5.0]  # High blur
        self.noise_range2 = [1, 5] # Low noise
        self.poisson_scale_range2 = [0.05, 0.5] # Low noise
        self.jpeg_range2 = [80, 95] # High quality
        
        self.gaussian_noise_prob = 0.5
        self.gray_noise_prob = 0.4
    
    def degrade(self, img_gt):
        """
        Apply Real-ESRGAN degradation
        
        Args:
            img_gt: torch tensor (C, H, W) in range [0, 1] (on GPU)
        
        Returns:
            img_lq: degraded tensor (on GPU)
        """
        img_gt = img_gt.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        device = img_gt.device # Get the device (e.g., 'cuda:0')
        
        ori_h, ori_w = img_gt.size()[2:4]
        
        # ----------------------- The first degradation process ----------------------- #
        
        # 1. BLUR
        # Applies a random blur kernel (Gaussian, anisotropic, etc.)
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma, # <-- Uses new [2.0, 8.0] range
            self.blur_sigma,
            [-np.pi, np.pi],
            self.betag_range, # <-- This will now work
            self.betap_range, # <-- This will now work
            noise_range=None
        )
        if isinstance(kernel, np.ndarray):
            kernel = torch.FloatTensor(kernel).to(device)
        img_lq = filter2D(img_gt, kernel)
        
        # 2. DOWNSAMPLING (Part 1: Random Resize)
        updown_type = np.random.choice(['up', 'down', 'keep'], p=[0.2, 0.7, 0.1])
        if updown_type == 'up':
            scale_factor = np.random.uniform(1, 1.5)
        elif updown_type == 'down':
            scale_factor = np.random.uniform(0.5, 1)
        else:
            scale_factor = 1
        
        if scale_factor != 1:
            img_lq = F_torch.interpolate(img_lq, scale_factor=scale_factor, mode='bilinear')
        
        # 3. NOISE
        # Adds either Gaussian or Poisson noise
        if np.random.uniform() < self.gaussian_noise_prob:
            img_lq = random_add_gaussian_noise_pt(
                img_lq, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob # <-- Uses new [1.0, 5.0] range
            )
        else:
            img_lq = random_add_poisson_noise_pt(
                img_lq,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False
            )
        
        # 4. JPEG COMPRESSION
        # Applies JPEG compression artifacts
        jpeg_p = img_lq.new_zeros(img_lq.size(0)).uniform_(*self.jpeg_range) # <-- Uses new [80, 95] range
        img_lq = torch.clamp(img_lq, 0, 1)
        # Move to CPU for JPEG compression (DiffJPEG works better on CPU)
        original_device = img_lq.device
        img_lq = self.jpeger(img_lq.cpu(), quality=jpeg_p).to(original_device)
        
        # ----------------------- The second degradation process ----------------------- #
        
        # 1. BLUR (Second Pass)
        if np.random.uniform() < self.second_blur_prob:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size2,
                self.blur_sigma2, # <-- Uses new [1.0, 5.0] range
                self.blur_sigma2,
                [-np.pi, np.pi],
                self.betag_range, # <-- This will now work
                self.betap_range, # <-- This will now work
                noise_range=None
            )
            if isinstance(kernel, np.ndarray):
                kernel = torch.FloatTensor(kernel).to(device)
            img_lq = filter2D(img_lq, kernel)
        
        # 2. DOWNSAMPLING (Part 2: Random Resize)
        updown_type = np.random.choice(['up', 'down', 'keep'], p=[0.3, 0.4, 0.3])
        if updown_type == 'up':
            scale_factor = np.random.uniform(1, 1.2)
        elif updown_type == 'down':
            scale_factor = np.random.uniform(0.5, 1)
        else:
            scale_factor = 1
        
        if scale_factor != 1:
            img_lq = F_torch.interpolate(img_lq, scale_factor=scale_factor, mode='bilinear')
        
        # 3. NOISE (Second Pass)
        if np.random.uniform() < self.gaussian_noise_prob:
            img_lq = random_add_gaussian_noise_pt(
                img_lq, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob # <-- Uses new [1.0, 5.0] range
            )
        else:
            img_lq = random_add_poisson_noise_pt(
                img_lq,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False
            )
        
        # 2. DOWNSAMPLING (Part 3: Final Resize)
        # This is the main downsampling step to the target 4x scale factor.
        # We use 'bilinear' or 'bicubic' to keep it smooth, as you requested.
        mode = np.random.choice(['bilinear', 'bicubic'])
        img_lq = F_torch.interpolate(
            img_lq, size=(ori_h // self.scale, ori_w // self.scale), mode=mode
        )
        
        # 4. JPEG COMPRESSION (Second Pass)
        jpeg_p = img_lq.new_zeros(img_lq.size(0)).uniform_(*self.jpeg_range2) # <-- Uses new [80, 95] range
        img_lq = torch.clamp(img_lq, 0, 1)
        # Move to CPU for JPEG compression (DiffJPEG works better on CPU)
        original_device = img_lq.device
        img_lq = self.jpeger(img_lq.cpu(), quality=jpeg_p).to(original_device)
        
        return img_lq.squeeze(0) # Squeeze batch dim


def process_dataset(hr_folder, output_base_dir, dataset_name, scale=4, patch_size=256, device='cpu'):
    """
    Process a dataset (train or valid) and generate patches.
    
    Args:
        hr_folder: Path to HR images folder
        output_base_dir: Base directory for output
        dataset_name: 'train' or 'valid'
        scale: Upscaling factor (default: 4)
        patch_size: Size of patches to extract (default: 256)
        device: Device to use ('cpu' or 'cuda')
    """
    # Create output folders
    hr_patches_folder = os.path.join(output_base_dir, f'DIV2K_{dataset_name}_HR_patches_256x256')
    lr_patches_folder = os.path.join(output_base_dir, f'DIV2K_{dataset_name}_LR_patches_256x256_upsampled')
    
    os.makedirs(hr_patches_folder, exist_ok=True)
    os.makedirs(lr_patches_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    print(f"HR folder: {hr_folder}")
    print(f"Output folders:")
    print(f"  - HR patches: {hr_patches_folder}")
    print(f"  - LR patches: {lr_patches_folder}\n")
    
    # Initialize degradation pipeline
    print("Initializing Real-ESRGAN degradation pipeline (SMOOTH BLUR)...")
    degrader = RealESRGANDegrader(scale=scale)
    # Don't move jpeger to device - it will handle device placement internally
    print("Pipeline ready!\n")
    
    # Get image paths
    hr_image_paths = sorted(glob.glob(os.path.join(hr_folder, '*.png')))
    
    if not hr_image_paths:
        print(f"ERROR: No images found in {hr_folder}")
        return 0
    
    print(f"Found {len(hr_image_paths)} images")
    print(f"Processing entire images on {str(device).upper()}")
    print(f"Extracting {patch_size}x{patch_size} patches after degradation")
    print(f"Upsampling LR patches back to {patch_size}x{patch_size}\n")
    
    patch_count = 0
    upsample_layer = torch.nn.Upsample(scale_factor=scale, mode='nearest').to(device)
    
    # Process each HR image
    for img_idx, img_path in enumerate(tqdm(hr_image_paths, desc=f"Processing {dataset_name} images")):
        try:
            # Load HR image
            img_hr_full = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_hr_full is None:
                print(f"Warning: Could not load {img_path}, skipping...")
                continue
            
            img_hr_full = img_hr_full.astype(np.float32) / 255.0
            img_hr_full = cv2.cvtColor(img_hr_full, cv2.COLOR_BGR2RGB)
            
            # Validate image values
            if np.any(np.isnan(img_hr_full)) or np.any(np.isinf(img_hr_full)):
                print(f"Warning: Invalid values in {img_path}, skipping...")
                continue
            
            # Ensure values are in valid range [0, 1]
            img_hr_full = np.clip(img_hr_full, 0.0, 1.0)
            
            h, w = img_hr_full.shape[:2]
            
            # Check image dimensions
            if h < patch_size or w < patch_size:
                print(f"Warning: Image {img_path} too small ({h}x{w}), skipping...")
                continue
            
            # Convert entire HR image to tensor and move to device
            hr_tensor_full = torch.from_numpy(np.transpose(img_hr_full, (2, 0, 1))).float().to(device)  # [C, H, W]
            
            # Validate tensor before processing
            if torch.any(torch.isnan(hr_tensor_full)) or torch.any(torch.isinf(hr_tensor_full)):
                print(f"Warning: Invalid tensor values in {img_path}, skipping...")
                continue
            
            # Apply Real-ESRGAN degradation to entire image
            with torch.no_grad():
                lr_tensor_full = degrader.degrade(hr_tensor_full)  # [C, H//4, W//4]
            
            # Validate degraded tensor
            if torch.any(torch.isnan(lr_tensor_full)) or torch.any(torch.isinf(lr_tensor_full)):
                print(f"Warning: Degradation produced invalid values for {img_path}, skipping...")
                continue
            
            # Upsample entire LR image back to HR size
            lr_tensor_upsampled = upsample_layer(lr_tensor_full.unsqueeze(0)).squeeze(0)  # [C, H, W]
            
            # Validate upsampled tensor
            if torch.any(torch.isnan(lr_tensor_upsampled)) or torch.any(torch.isinf(lr_tensor_upsampled)):
                print(f"Warning: Upsampling produced invalid values for {img_path}, skipping...")
                continue
            
            # Move back to CPU for patch extraction
            hr_full_cpu = hr_tensor_full.cpu().numpy()
            lr_full_cpu = lr_tensor_upsampled.cpu().numpy()
            
            # Extract non-overlapping patches
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            
            # Prepare batch of patches for saving
            hr_patches_to_save = []
            lr_patches_to_save = []
            patch_names = []
            
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    # Extract patch coordinates
                    y_start = i * patch_size
                    x_start = j * patch_size
                    y_end = y_start + patch_size
                    x_end = x_start + patch_size
                    
                    # Extract patches from numpy arrays [C, H, W] -> [H, W, C]
                    hr_patch_np = np.transpose(hr_full_cpu[:, y_start:y_end, x_start:x_end], (1, 2, 0))
                    lr_patch_np = np.transpose(lr_full_cpu[:, y_start:y_end, x_start:x_end], (1, 2, 0))
                    
                    # Clip and convert to uint8
                    hr_patch_np = np.clip(hr_patch_np * 255.0, 0, 255).astype(np.uint8)
                    lr_patch_np = np.clip(lr_patch_np * 255.0, 0, 255).astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV
                    hr_patch_bgr = cv2.cvtColor(hr_patch_np, cv2.COLOR_RGB2BGR)
                    lr_patch_bgr = cv2.cvtColor(lr_patch_np, cv2.COLOR_RGB2BGR)
                    
                    # Store for batch saving
                    hr_patches_to_save.append(hr_patch_bgr)
                    lr_patches_to_save.append(lr_patch_bgr)
                    
                    basename = os.path.splitext(os.path.basename(img_path))[0]
                    patch_names.append(f"{basename}_patch_{i}_{j}.png")
            
            # Batch save all patches for this image
            for idx, patch_name in enumerate(patch_names):
                hr_patch_path = os.path.join(hr_patches_folder, patch_name)
                lr_patch_path = os.path.join(lr_patches_folder, patch_name)
                cv2.imwrite(hr_patch_path, hr_patches_to_save[idx])
                cv2.imwrite(lr_patch_path, lr_patches_to_save[idx])
                patch_count += 1
        
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{dataset_name.upper()} Dataset Complete!")
    print(f"  - Processed {len(hr_image_paths)} images")
    print(f"  - Generated {patch_count} patch pairs")
    print(f"  - HR patches: {hr_patches_folder}")
    print(f"  - LR patches: {lr_patches_folder}\n")
    
    return patch_count


def main():
    """Main function to process both training and validation datasets"""
    # Configuration - use paths from config
    from config import _project_root, scale, patch_size
    
    # Force CPU usage
    device = torch.device("cpu")
    print("="*60)
    print("DiffusionSR Patch Generation (CPU Mode)")
    print("="*60)
    print(f"Using device: {device}")
    print(f"Scale factor: {scale}x")
    print(f"Patch size: {patch_size}x{patch_size}\n")
    
    # Dataset paths
    data_dir = os.path.join(_project_root, 'data')
    train_hr_folder = os.path.join(data_dir, 'DIV2K_train_HR')
    valid_hr_folder = os.path.join(data_dir, 'DIV2K_valid_HR')
    
    # Output base directory
    output_base_dir = data_dir
    
    total_train_patches = 0
    total_valid_patches = 0
    
    # Process training dataset
    if os.path.exists(train_hr_folder):
        total_train_patches = process_dataset(
            hr_folder=train_hr_folder,
            output_base_dir=output_base_dir,
            dataset_name='train',
            scale=scale,
            patch_size=patch_size,
            device=device
        )
    else:
        print(f"WARNING: Training folder not found: {train_hr_folder}\n")
    
    # Process validation dataset
    if os.path.exists(valid_hr_folder):
        total_valid_patches = process_dataset(
            hr_folder=valid_hr_folder,
            output_base_dir=output_base_dir,
            dataset_name='valid',
            scale=scale,
            patch_size=patch_size,
            device=device
        )
    else:
        print(f"WARNING: Validation folder not found: {valid_hr_folder}\n")
    
    # Summary
    print("="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Training patches: {total_train_patches:,}")
    print(f"Validation patches: {total_valid_patches:,}")
    print(f"Total patches: {total_train_patches + total_valid_patches:,}")
    
    # Display sample patches from training set
    train_hr_patches_folder = os.path.join(output_base_dir, 'DIV2K_train_HR_patches_256x256')
    train_lr_patches_folder = os.path.join(output_base_dir, 'DIV2K_train_LR_patches_256x256_upsampled')
    
    sample_patches = sorted(glob.glob(os.path.join(train_hr_patches_folder, '*.png')))[:5]
    if sample_patches:
        print("\nDisplaying sample patches from training set...")
        fig, axes = plt.subplots(len(sample_patches), 2, figsize=(10, len(sample_patches) * 2))
        if len(sample_patches) == 1:
            axes = np.array([axes])
        
        for i, hr_patch_path in enumerate(sample_patches):
            basename = os.path.basename(hr_patch_path)
            lr_patch_path = os.path.join(train_lr_patches_folder, basename)
            
            if os.path.exists(lr_patch_path):
                hr = cv2.imread(hr_patch_path)
                lr = cv2.imread(lr_patch_path)
                
                hr_rgb = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
                lr_rgb = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
                
                axes[i, 0].imshow(hr_rgb)
                axes[i, 0].set_title(f"HR Patch: {basename}", fontweight='bold')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(lr_rgb)
                axes[i, 1].set_title(f"LR Patch (upsampled): {basename}", fontweight='bold')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(_project_root, 'data', 'sample_patches.png'), dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to: {os.path.join(_project_root, 'data', 'sample_patches.png')}")
    
    print("\nDone! Dataset generation complete.")
    print(f"\nNext steps:")
    print(f"   1. Update config.py:")
    print(f"      - Set dir_HR = '{train_hr_patches_folder}'")
    print(f"      - Set dir_LR = '{train_lr_patches_folder}'")
    print(f"   2. The SRDataset will now:")
    print(f"      - Load pre-generated 256x256 HR patches")
    print(f"      - Load pre-generated 256x256 upsampled LR patches")
    print(f"      - Skip cropping (patches are already the right size)")
    print(f"      - Apply augmentations (flip, rotate)")
    print(f"   3. Training will use these patches directly (no upsampling needed)")

if __name__ == "__main__":
    main()