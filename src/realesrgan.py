import os
import sys
import glob
import cv2
import numpy as np
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# Apply compatibility patch BEFORE importing basicsr
# This fixes the issue where basicsr tries to import torchvision.transforms.functional_tensor
# which doesn't exist in newer torchvision versions
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except:
    pass

# Import Real-ESRGAN's actual degradation pipeline
from basicsr.data.degradations import (
    random_add_gaussian_noise_pt, 
    random_add_poisson_noise_pt,
    random_mixed_kernels,
    circular_lowpass_kernel
)
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from torch.nn import functional as F_torch

class RealESRGANDegrader:
    """Real-ESRGAN degradation pipeline matching original ResShift implementation"""
    
    def __init__(self, scale=4):
        self.scale = scale
        
        # Initialize JPEG compression
        self.jpeger = DiffJPEG(differentiable=False)
        
        # Import all parameters from config
        from config import (
            blur_kernel_size, kernel_list, kernel_prob,
            data_train_blur_sigma as blur_sigma,
            noise_range, poisson_scale_range, jpeg_range,
            data_train_blur_sigma2 as blur_sigma2,
            noise_range2, poisson_scale_range2, jpeg_range2,
            second_order_prob, second_blur_prob, final_sinc_prob,
            resize_prob, resize_range, resize_prob2, resize_range2,
            gaussian_noise_prob, gray_noise_prob, gaussian_noise_prob2, gray_noise_prob2,
            data_train_betag_range as betag_range,
            data_train_betap_range as betap_range,
            data_train_betag_range2 as betag_range2,
            data_train_betap_range2 as betap_range2,
            data_train_blur_kernel_size2 as blur_kernel_size2,
            data_train_sinc_prob as sinc_prob,
            data_train_sinc_prob2 as sinc_prob2
        )
        
        # Blur kernel settings
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        
        # First degradation parameters
        self.blur_sigma = blur_sigma
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.jpeg_range = jpeg_range
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob
        
        # Second degradation parameters
        self.second_order_prob = second_order_prob
        self.second_blur_prob = second_blur_prob
        self.blur_kernel_size2 = blur_kernel_size2
        self.blur_sigma2 = blur_sigma2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.jpeg_range2 = jpeg_range2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2
        
        # Final sinc filter
        self.final_sinc_prob = final_sinc_prob
        
        # Resize parameters
        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2
        
        # Noise probabilities
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gray_noise_prob = gray_noise_prob
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.gray_noise_prob2 = gray_noise_prob2
        
        # Kernel ranges for sinc filter generation
        self.kernel_range1 = [x for x in range(3, self.blur_kernel_size, 2)]
        self.kernel_range2 = [x for x in range(3, self.blur_kernel_size2, 2)]
        
        # Pulse tensor (identity kernel) for final sinc filter
        self.pulse_tensor = torch.zeros(self.blur_kernel_size2, self.blur_kernel_size2).float()
        self.pulse_tensor[self.blur_kernel_size2//2, self.blur_kernel_size2//2] = 1
    
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
        
        # 2. RANDOM RESIZE (First degradation)
        updown_type = random.choices(['up', 'down', 'keep'], weights=self.resize_prob)[0]
        if updown_type == 'up':
            scale_factor = random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale_factor = random.uniform(self.resize_range[0], 1)
        else:
            scale_factor = 1
        
        if scale_factor != 1:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            img_lq = F_torch.interpolate(img_lq, scale_factor=scale_factor, mode=mode)
        
        # 3. NOISE (First degradation)
        if random.random() < self.gaussian_noise_prob:
            img_lq = random_add_gaussian_noise_pt(
                img_lq, 
                sigma_range=self.noise_range, 
                clip=True, 
                rounds=False, 
                gray_prob=self.gray_noise_prob
            )
        else:
            img_lq = random_add_poisson_noise_pt(
                img_lq,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False
            )
        
        # 4. JPEG COMPRESSION (First degradation)
        jpeg_p = img_lq.new_zeros(img_lq.size(0)).uniform_(*self.jpeg_range)
        img_lq = torch.clamp(img_lq, 0, 1)
        original_device = img_lq.device
        img_lq = self.jpeger(img_lq.cpu(), quality=jpeg_p).to(original_device)
        
        # ----------------------- The second degradation process (50% probability) ----------------------- #
        
        if random.random() < self.second_order_prob:
            # 1. BLUR (Second Pass)
            if random.random() < self.second_blur_prob:
                # Generate second kernel
                kernel_size2 = random.choice(self.kernel_range2)
                if random.random() < self.sinc_prob2:
                    # Sinc kernel for second degradation
                    if kernel_size2 < 13:
                        omega_c = random.uniform(math.pi / 3, math.pi)
                    else:
                        omega_c = random.uniform(math.pi / 5, math.pi)
                    kernel2 = circular_lowpass_kernel(omega_c, kernel_size2, pad_to=False)
                else:
                    kernel2 = random_mixed_kernels(
                        self.kernel_list,
                        self.kernel_prob,
                        kernel_size2,
                        self.blur_sigma2,
                        self.blur_sigma2,
                        [-math.pi, math.pi],
                        self.betag_range2,
                        self.betap_range2,
                        noise_range=None
                    )
                # Pad kernel
                pad_size = (self.blur_kernel_size2 - kernel_size2) // 2
                kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
                if isinstance(kernel2, np.ndarray):
                    kernel2 = torch.FloatTensor(kernel2).to(device)
                img_lq = filter2D(img_lq, kernel2)
            
            # 2. RANDOM RESIZE (Second degradation)
            updown_type = random.choices(['up', 'down', 'keep'], weights=self.resize_prob2)[0]
            if updown_type == 'up':
                scale_factor = random.uniform(1, self.resize_range2[1])
            elif updown_type == 'down':
                scale_factor = random.uniform(self.resize_range2[0], 1)
            else:
                scale_factor = 1
            
            if scale_factor != 1:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                img_lq = F_torch.interpolate(
                    img_lq,
                    size=(int(ori_h / self.scale * scale_factor), int(ori_w / self.scale * scale_factor)),
                    mode=mode
                )
            
            # 3. NOISE (Second Pass)
            if random.random() < self.gaussian_noise_prob2:
                img_lq = random_add_gaussian_noise_pt(
                    img_lq, 
                    sigma_range=self.noise_range2, 
                    clip=True, 
                    rounds=False, 
                    gray_prob=self.gray_noise_prob2
                )
            else:
                img_lq = random_add_poisson_noise_pt(
                    img_lq,
                    scale_range=self.poisson_scale_range2,
                    gray_prob=self.gray_noise_prob2,
                    clip=True,
                    rounds=False
                )
        
        # ----------------------- Final stage: Resize back + Sinc filter + JPEG ----------------------- #
        
        # Generate final sinc kernel
        if random.random() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = random.uniform(math.pi / 3, math.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel).to(device)
        else:
            sinc_kernel = self.pulse_tensor.to(device)  # Identity (no sinc filter)
        
        # Randomize order: [resize + sinc] + JPEG vs JPEG + [resize + sinc]
        if random.random() < 0.5:
            # Order 1: Resize back + sinc filter, then JPEG
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            img_lq = F_torch.interpolate(
                img_lq,
                size=(ori_h // self.scale, ori_w // self.scale),
                mode=mode
            )
            img_lq = filter2D(img_lq, sinc_kernel)
            # JPEG compression
            jpeg_p = img_lq.new_zeros(img_lq.size(0)).uniform_(*self.jpeg_range2)
            img_lq = torch.clamp(img_lq, 0, 1)
            original_device = img_lq.device
            img_lq = self.jpeger(img_lq.cpu(), quality=jpeg_p).to(original_device)
        else:
            # Order 2: JPEG compression, then resize back + sinc filter
            jpeg_p = img_lq.new_zeros(img_lq.size(0)).uniform_(*self.jpeg_range2)
            img_lq = torch.clamp(img_lq, 0, 1)
            original_device = img_lq.device
            img_lq = self.jpeger(img_lq.cpu(), quality=jpeg_p).to(original_device)
            # Resize back + sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            img_lq = F_torch.interpolate(
                img_lq,
                size=(ori_h // self.scale, ori_w // self.scale),
                mode=mode
            )
            img_lq = filter2D(img_lq, sinc_kernel)
        
        # Clamp and round (final step)
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.0
        
        return img_lq.squeeze(0)  # Squeeze batch dim


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