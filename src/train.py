"""
Main training script for ResShift diffusion model.

This script initializes the Trainer class and runs the main training loop.
"""

import multiprocessing
# Fix CUDA multiprocessing: Set start method to 'spawn' for compatibility with CUDA
# This is required when using DataLoader with num_workers > 0 on systems where
# CUDA is initialized before worker processes are created (Colab, some Linux setups)
# Must be set before any CUDA initialization or DataLoader creation
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method already set (e.g., in another module), ignore
    pass

from trainer import Trainer
from config import (
    iterations, batch_size, microbatch, learning_rate,
    warmup_iterations, save_freq, log_freq, T, k, val_freq
)
import torch
import wandb


def train(resume_ckpt=None):
    """
    Main training loop that integrates all components.
    
    Training flow:
    1. Build model and dataloader
    2. Setup optimization
    3. Training loop:
       - Get batch from dataloader
       - Training step (forward, backward, optimizer step)
       - Adjust learning rate
       - Log metrics and images
       - Save checkpoints
    
    Args:
        resume_ckpt: Path to checkpoint file to resume from (optional)
    """
    # Initialize trainer
    trainer = Trainer(resume_ckpt=resume_ckpt)
    
    print("=" * 100)
    if resume_ckpt:
        print("Resuming Training")
    else:
        print("Starting Training")
    print("=" * 100)
    
    # Build model (Component 2)
    trainer.build_model()
    
    # Resume from checkpoint if provided (must be after model is built)
    if resume_ckpt:
        trainer.resume_from_ckpt(resume_ckpt)
    
    # Setup optimization (Component 1)
    trainer.setup_optimization()
    
    # Build dataloader (Component 3)
    trainer.build_dataloader()
    
    # Initialize training
    trainer.model.train()
    train_iter = iter(trainer.dataloaders['train'])
    
    print(f"\nTraining Configuration:")
    print(f"  - Total iterations: {iterations}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Micro-batch size: {microbatch}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Warmup iterations: {warmup_iterations}")
    print(f"  - Save frequency: {save_freq}")
    print(f"  - Log frequency: {log_freq}")
    print(f"  - Device: {trainer.device}")
    print("=" * 100)
    print("\nStarting training loop...\n")
    
    # Training loop
    for step in range(trainer.iters_start, iterations):
        trainer.current_iters = step + 1
        
        # Get batch from dataloader
        try:
            hr_latent, lr_latent = next(train_iter)
        except StopIteration:
            # Restart iterator if exhausted (shouldn't happen with infinite cycle, but safety)
            train_iter = iter(trainer.dataloaders['train'])
            hr_latent, lr_latent = next(train_iter)
        
        # Move to device
        hr_latent = hr_latent.to(trainer.device)
        lr_latent = lr_latent.to(trainer.device)
        
        # Training step (Component 5)
        # This handles: forward pass, backward pass, optimizer step, gradient accumulation
        loss, timing_dict = trainer.training_step(hr_latent, lr_latent)
        
        # Adjust learning rate (Component 6)
        trainer.adjust_lr()
        
        # Run validation (Component 9)
        if 'val' in trainer.dataloaders and trainer.current_iters % val_freq == 0:
            trainer.validation()
        
        # Prepare data for logging (need x_t and pred for visualization)
        # Recompute for logging (or store in training_step - for now, recompute)
        with torch.no_grad():
            residual = (lr_latent - hr_latent)
            t_log = torch.randint(0, T, (hr_latent.shape[0],)).to(trainer.device)
            epsilon_log = torch.randn_like(hr_latent)
            eta_t_log = trainer.eta[t_log]
            x_t_log = hr_latent + eta_t_log * residual + k * torch.sqrt(eta_t_log) * epsilon_log
            
            trainer.model.eval()
            # Model predicts x0 (clean HR latent), not noise
            x0_pred_log = trainer.model(x_t_log[0:1], t_log[0:1], lq=lr_latent[0:1])
            trainer.model.train()
        
        # Store timing info for logging
        trainer._last_timing = timing_dict
        
        # Log training metrics and images (Component 8)
        trainer.log_step_train(
            loss=loss,
            hr_latent=hr_latent[0:1],
            lr_latent=lr_latent[0:1],
            x_t=x_t_log[0:1],
            pred=x0_pred_log,  # x0 prediction (clean HR latent)
            phase='train'
        )
        
        # Save checkpoint (Component 7)
        if trainer.current_iters % save_freq == 0:
            trainer.save_ckpt()
    
    # Final checkpoint
    print("\n" + "=" * 100)
    print("Training completed!")
    print("=" * 100)
    trainer.save_ckpt()
    print(f"Final checkpoint saved at iteration {trainer.current_iters}")
    
    # Finish WandB
    wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResShift diffusion model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume from (e.g., checkpoints/ckpts/model_10000.pth)')
    
    args = parser.parse_args()
    
    train(resume_ckpt=args.resume)
