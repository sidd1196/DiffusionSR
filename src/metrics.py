"""
Metrics for image quality evaluation.

This module provides PSNR, SSIM, and LPIPS metrics for evaluating
super-resolution model performance.
"""

import torch
import lpips


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images.
    
    Args:
        img1: First image tensor (B, C, H, W) in range [0, 1]
        img2: Second image tensor (B, C, H, W) in range [0, 1]
    
    Returns:
        psnr: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1, img2):
    """
    Compute SSIM between two images.
    
    Args:
        img1: First image tensor (B, C, H, W) in range [0, 1]
        img2: Second image tensor (B, C, H, W) in range [0, 1]
    
    Returns:
        ssim: SSIM value (0 to 1, higher is better)
    """
    # SSIM parameters
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = torch.mean(img1, dim=[2, 3], keepdim=True)
    mu2 = torch.mean(img2, dim=[2, 3], keepdim=True)
    
    sigma1_sq = torch.var(img1, dim=[2, 3], keepdim=True)
    sigma2_sq = torch.var(img2, dim=[2, 3], keepdim=True)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[2, 3], keepdim=True)
    
    ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = ssim_n / ssim_d
    return ssim.mean().item()


def compute_lpips(img1, img2, lpips_model):
    """
    Compute LPIPS between two images.
    
    Args:
        img1: First image tensor (B, C, H, W) in range [0, 1]
        img2: Second image tensor (B, C, H, W) in range [0, 1]
        lpips_model: Initialized LPIPS model
    
    Returns:
        lpips: LPIPS value (lower is better, typically 0-1 range)
    """
    # LPIPS expects images in range [-1, 1]
    img1_lpips = img1 * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    img2_lpips = img2 * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    
    # Ensure tensors are on the correct device
    device = next(lpips_model.parameters()).device
    img1_lpips = img1_lpips.to(device)
    img2_lpips = img2_lpips.to(device)
    
    with torch.no_grad():
        lpips_value = lpips_model(img1_lpips, img2_lpips).mean().item()
    
    return lpips_value
