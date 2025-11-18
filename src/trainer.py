import torch
import torch.nn as nn
import random
import numpy as np
from model import FullUNET
from noiseControl import resshift_schedule
from torch.utils.data import DataLoader
from data import mini_dataset, train_dataset, valid_dataset, get_vqgan_model
import torch.optim as optim
from config import (
    batch_size, device, learning_rate, iterations, 
    weight_decay, T, k, _project_root, num_workers,
    use_amp, lr, lr_min, lr_schedule, warmup_iterations,
    compile_flag, compile_mode, batch, prefetch_factor, microbatch,
    save_freq, log_freq, val_freq, val_y_channel, ema_rate, use_ema_val,
    seed, global_seeding, normalize_input, latent_flag
)
import wandb
import os
import math
import time
from pathlib import Path
from itertools import cycle
from contextlib import nullcontext
from dotenv import load_dotenv
from metrics import compute_psnr, compute_ssim, compute_lpips
from ema import EMA
import lpips


class Trainer:
    """
    Modular trainer class following the original ResShift trainer structure.
    """
    
    def __init__(self, save_dir=None, resume_ckpt=None):
        """
        Initialize trainer with config values.
        
        Args:
            save_dir: Directory to save checkpoints (defaults to _project_root / 'checkpoints')
            resume_ckpt: Path to checkpoint file to resume from (optional)
        """
        self.device = device
        self.current_iters = 0
        self.iters_start = 0
        self.resume_ckpt = resume_ckpt
        
        # Setup checkpoint directory
        if save_dir is None:
            save_dir = _project_root / 'checkpoints'
        self.save_dir = Path(save_dir)
        self.ckpt_dir = self.save_dir / 'ckpts'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize noise schedule (eta values for ResShift)
        self.eta = resshift_schedule().to(self.device)
        self.eta = self.eta[:, None, None, None]  # shape (T, 1, 1, 1)
        
        # Loss criterion
        self.criterion = nn.MSELoss()
        
        # Timing for checkpoint saving
        self.tic = None
        
        # EMA will be initialized after model is built
        self.ema = None
        self.ema_model = None
        
        # Set random seeds for reproducibility
        self.setup_seed()
        
        # Initialize WandB
        self.init_wandb()
        
    def setup_seed(self, seed_val=None, global_seeding_val=None):
        """
        Set random seeds for reproducibility.
        
        Sets seeds for:
        - Python random module
        - NumPy
        - PyTorch (CPU and CUDA)
        
        Args:
            seed_val: Seed value (defaults to config.seed)
            global_seeding_val: Whether to use global seeding (defaults to config.global_seeding)
        """
        if seed_val is None:
            seed_val = seed
        if global_seeding_val is None:
            global_seeding_val = global_seeding
        
        # Set Python random seed
        random.seed(seed_val)
        
        # Set NumPy random seed
        np.random.seed(seed_val)
        
        # Set PyTorch random seed
        torch.manual_seed(seed_val)
        
        # Set CUDA random seeds (if available)
        if torch.cuda.is_available():
            if global_seeding_val:
                torch.cuda.manual_seed_all(seed_val)
            else:
                torch.cuda.manual_seed(seed_val)
                # For multi-GPU, each GPU would get seed + rank (not implemented here)
        
        # Make deterministic (may impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
        print(f"✓ Random seeds set: seed={seed_val}, global_seeding={global_seeding_val}")
    
    def init_wandb(self):
        """Initialize WandB logging."""
        load_dotenv(os.path.join(_project_root, '.env'))
        wandb.init(
            project="diffusionsr",
            name="reshift_training",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "steps": iterations,
                "model": "ResShift",
                "T": T,
                "k": k,
                "optimizer": "AdamW" if weight_decay > 0 else "Adam",
                "betas": (0.9, 0.999),
                "grad_clip": 1.0,
                "criterion": "MSE",
                "device": str(device),
                "training_space": "latent_64x64",
                "use_amp": use_amp,
                "ema_rate": 0.999 if hasattr(self, 'ema_rate') else None
            }
        )
    
    def setup_optimization(self):
        """
        Component 1: Setup optimizer and AMP scaler.
        
        Sets up:
        - Optimizer (AdamW with weight decay or Adam)
        - AMP GradScaler if use_amp is True
        """
        # Use AdamW if weight_decay > 0, otherwise Adam
        if weight_decay > 0:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        
        # AMP settings: Create GradScaler if use_amp is True and CUDA is available
        if use_amp and torch.cuda.is_available():
            self.amp_scaler = torch.amp.GradScaler('cuda')
        else:
            self.amp_scaler = None
            if use_amp and not torch.cuda.is_available():
                print("  ⚠ Warning: AMP requested but CUDA not available. Disabling AMP.")
        
        # Learning rate scheduler (cosine annealing after warmup)
        self.lr_scheduler = None
        if lr_schedule == 'cosin':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=iterations - warmup_iterations,
                eta_min=lr_min
            )
            print(f"  - LR scheduler: CosineAnnealingLR (T_max={iterations - warmup_iterations}, eta_min={lr_min})")
        
        # Load pending optimizer state if resuming
        if hasattr(self, '_pending_optimizer_state'):
            self.optimizer.load_state_dict(self._pending_optimizer_state)
            print(f"  - Loaded optimizer state from checkpoint")
            delattr(self, '_pending_optimizer_state')
        
        # Load pending LR scheduler state if resuming
        if hasattr(self, '_pending_lr_scheduler_state') and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(self._pending_lr_scheduler_state)
            print(f"  - Loaded LR scheduler state from checkpoint")
            delattr(self, '_pending_lr_scheduler_state')
        
        # Restore LR schedule by replaying adjust_lr for all previous iterations
        # This ensures the LR is at the correct value for the resumed iteration
        if hasattr(self, '_resume_iters') and self._resume_iters > 0:
            print(f"  - Restoring learning rate schedule to iteration {self._resume_iters}...")
            for ii in range(1, self._resume_iters + 1):
                self.adjust_lr(ii)
            print(f"  - ✓ Learning rate schedule restored")
            delattr(self, '_resume_iters')
        
        print(f"✓ Setup optimization:")
        print(f"  - Optimizer: {type(self.optimizer).__name__}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Warmup iterations: {warmup_iterations}")
        print(f"  - LR schedule: {lr_schedule if lr_schedule else 'None (fixed LR)'}")
        print(f"  - AMP enabled: {use_amp} ({'GradScaler active' if self.amp_scaler else 'disabled'})")
    
    def build_model(self):
        """
        Component 2: Build model and autoencoder (VQGAN).
        
        Sets up:
        - FullUNET model
        - Model compilation (optional)
        - VQGAN autoencoder for encoding/decoding
        - Model info printing
        """
        # Build main model
        print("Building FullUNET model...")
        self.model = FullUNET()
        self.model = self.model.to(self.device)
        
        # Optional: Compile model for optimization
        # Model compilation can provide 20-30% speedup on modern GPUs
        # but requires PyTorch 2.0+ and may have compatibility issues
        self.model_compiled = False
        if compile_flag:
            try:
                print(f"Compiling model with mode: {compile_mode}...")
                self.model = torch.compile(self.model, mode=compile_mode)
                self.model_compiled = True
                print("✓ Model compilation done")
            except Exception as e:
                print(f"⚠ Warning: Model compilation failed: {e}")
                print("  Continuing without compilation...")
                self.model_compiled = False
        
        # Load VQGAN autoencoder
        print("Loading VQGAN autoencoder...")
        self.autoencoder = get_vqgan_model()
        print("✓ VQGAN autoencoder loaded")
        
        # Initialize LPIPS model for validation
        print("Loading LPIPS metric...")
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        for params in self.lpips_model.parameters():
            params.requires_grad_(False)
        self.lpips_model.eval()
        print("✓ LPIPS metric loaded")
        
        # Initialize EMA if enabled
        if ema_rate > 0:
            print(f"Initializing EMA with rate: {ema_rate}...")
            self.ema = EMA(self.model, ema_rate=ema_rate, device=self.device)
            # Add Swin Transformer relative position index to ignore keys
            self.ema.add_ignore_key('relative_position_index')
            print("✓ EMA initialized")
        else:
            print("⚠ EMA disabled (ema_rate = 0)")
        
        # Print model information
        self.print_model_info()
    
    def print_model_info(self):
        """Print model parameter count and architecture info."""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n✓ Model built successfully:")
        print(f"  - Model: FullUNET")
        print(f"  - Total parameters: {total_params / 1e6:.2f}M")
        print(f"  - Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"  - Device: {self.device}")
        print(f"  - Compiled: {'Yes' if getattr(self, 'model_compiled', False) else 'No'}")
        if self.autoencoder is not None:
            print(f"  - Autoencoder: VQGAN (loaded)")
    
    def build_dataloader(self):
        """
        Component 3: Build train and validation dataloaders.
        
        Sets up:
        - Train dataloader with infinite cycle wrapper
        - Validation dataloader (if validation dataset exists)
        - Proper batch sizes, num_workers, pin_memory, etc.
        """
        def _wrap_loader(loader):
            """Wrap dataloader to cycle infinitely."""
            while True:
                yield from loader
        
        # Create datasets dictionary
        datasets = {'train': train_dataset}
        if valid_dataset is not None:
            datasets['val'] = valid_dataset
        
        # Print dataset sizes
        for phase, dataset in datasets.items():
            print(f"  - {phase.capitalize()} dataset: {len(dataset)} images")
        
        # Create train dataloader
        train_batch_size = batch[0]  # Use first value from batch list
        train_loader = DataLoader(
            datasets['train'],
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,  # Drop last incomplete batch
            num_workers=min(num_workers, 4),  # Limit num_workers
            pin_memory=True if torch.cuda.is_available() else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        
        # Wrap train loader to cycle infinitely
        self.dataloaders = {'train': _wrap_loader(train_loader)}
        
        # Create validation dataloader if validation dataset exists
        if 'val' in datasets:
            val_batch_size = batch[1] if len(batch) > 1 else batch[0]  # Use second value or fallback
            val_loader = DataLoader(
                datasets['val'],
                batch_size=val_batch_size,
                shuffle=False,
                drop_last=False,  # Don't drop last batch in validation
                num_workers=0,  # No multiprocessing for validation (safer)
                pin_memory=True if torch.cuda.is_available() else False,
            )
            self.dataloaders['val'] = val_loader
        
        # Store datasets
        self.datasets = datasets
        
        print(f"\n✓ Dataloaders built:")
        print(f"  - Train batch size: {train_batch_size}")
        print(f"  - Train num_workers: {min(num_workers, 4)}")
        print(f"  - Train drop_last: True")
        if 'val' in self.dataloaders:
            print(f"  - Val batch size: {val_batch_size}")
            print(f"  - Val num_workers: 0")
    
    def backward_step(self, loss, num_grad_accumulate=1):
        """
        Component 4: Handle backward pass with AMP support and gradient accumulation.
        
        Args:
            loss: The computed loss tensor
            num_grad_accumulate: Number of gradient accumulation steps (for micro-batching)
        
        Returns:
            loss: The loss tensor (for logging)
        """
        # Normalize loss by gradient accumulation steps
        loss = loss / num_grad_accumulate
        
        # Backward pass: use AMP scaler if available, otherwise direct backward
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()
        
        return loss
    
    def _scale_input(self, x_t, t):
        """
        Scale input based on timestep for training stability.
        Matches original GaussianDiffusion._scale_input for latent space.
        
        For latent space: std = sqrt(etas[t] * kappa^2 + 1)
        This normalizes the input variance across different timesteps.
        
        Args:
            x_t: Noisy input tensor (B, C, H, W)
            t: Timestep tensor (B,)
        
        Returns:
            x_t_scaled: Scaled input tensor (B, C, H, W)
        """
        if normalize_input and latent_flag:
            # For latent space: std = sqrt(etas[t] * kappa^2 + 1)
            # Extract eta_t for each sample in batch
            eta_t = self.eta[t]  # (B, 1, 1, 1)
            std = torch.sqrt(eta_t * k**2 + 1)
            x_t_scaled = x_t / std
        else:
            x_t_scaled = x_t
        return x_t_scaled
    
    def training_step(self, hr_latent, lr_latent):
        """
        Component 5: Main training step with micro-batching and gradient accumulation.
        
        Args:
            hr_latent: High-resolution latent tensor (B, C, 64, 64)
            lr_latent: Low-resolution latent tensor (B, C, 64, 64)
        
        Returns:
            loss: Average loss value for logging
            timing_dict: Dictionary with timing information
        """
        step_start = time.time()
        
        self.model.train()
        
        current_batchsize = hr_latent.shape[0]
        micro_batchsize = microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)
        
        total_loss = 0.0
        
        forward_time = 0.0
        backward_time = 0.0
        
        # Process in micro-batches for gradient accumulation
        for jj in range(0, current_batchsize, micro_batchsize):
            # Extract micro-batch
            end_idx = min(jj + micro_batchsize, current_batchsize)
            hr_micro = hr_latent[jj:end_idx].to(self.device)
            lr_micro = lr_latent[jj:end_idx].to(self.device)
            last_batch = (end_idx >= current_batchsize)
            
            # Compute residual in latent space
            residual = (lr_micro - hr_micro)
            
            # Generate random timesteps for each sample in micro-batch
            t = torch.randint(0, T, (hr_micro.shape[0],)).to(self.device)
            
            # Add noise in latent space (ResShift noise schedule)
            epsilon = torch.randn_like(hr_micro)  # Noise in latent space
            eta_t = self.eta[t]  # (B, 1, 1, 1)
            x_t = hr_micro + eta_t * residual + k * torch.sqrt(eta_t) * epsilon
            
            # Forward pass with autocast if AMP is enabled
            forward_start = time.time()
            if use_amp and torch.cuda.is_available():
                context = torch.amp.autocast('cuda')
            else:
                context = nullcontext()
            with context:
                # Scale input for training stability (normalize variance across timesteps)
                x_t_scaled = self._scale_input(x_t, t)
                # Forward pass: Model predicts x0 (clean HR latent), not noise
                # ResShift uses predict_type = "xstart"
                x0_pred = self.model(x_t_scaled, t, lq=lr_micro)
                # Loss: Compare predicted x0 with ground truth HR latent
                loss = self.criterion(x0_pred, hr_micro)
            forward_time += time.time() - forward_start
            
            # Store loss value for logging (before dividing for gradient accumulation)
            total_loss += loss.item()
            
            # Backward step (handles gradient accumulation and AMP)
            backward_start = time.time()
            self.backward_step(loss, num_grad_accumulate)
            backward_time += time.time() - backward_start
        
        # Gradient clipping before optimizer step
        if self.amp_scaler is None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        else:
            # Unscale gradients before clipping when using AMP
            self.amp_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Update EMA after optimizer step
        if self.ema is not None:
            self.ema.update(self.model)
        
        # Compute total step time
        step_time = time.time() - step_start
        
        # Return average loss (average across micro-batches)
        num_micro_batches = math.ceil(current_batchsize / micro_batchsize)
        avg_loss = total_loss / num_micro_batches if num_micro_batches > 0 else total_loss
        
        # Return timing information
        timing_dict = {
            'step_time': step_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'num_micro_batches': num_micro_batches
        }
        
        return avg_loss, timing_dict
    
    def adjust_lr(self, current_iters=None):
        """
        Component 6: Adjust learning rate with warmup and optional cosine annealing.
        
        Learning rate schedule:
        - Warmup phase (iters <= warmup_iterations): Linear increase from 0 to base_lr
        - After warmup: Use cosine annealing scheduler if lr_schedule == 'cosin', else keep base_lr
        
        Args:
            current_iters: Current iteration number (defaults to self.current_iters)
        """
        base_lr = learning_rate
        warmup_steps = warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        
        if current_iters <= warmup_steps:
            # Warmup phase: linear increase from 0 to base_lr
            warmup_lr = (current_iters / warmup_steps) * base_lr
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = warmup_lr
        else:
            # After warmup: use scheduler if available
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
    def save_ckpt(self):
        """
        Component 7: Save checkpoint with model state, optimizer state, and training info.
        
        Saves:
        - Model state dict
        - Optimizer state dict
        - Current iteration number
        - AMP scaler state (if AMP is enabled)
        - LR scheduler state (if scheduler exists)
        """
        ckpt_path = self.ckpt_dir / f'model_{self.current_iters}.pth'
        
        # Prepare checkpoint dictionary
        ckpt = {
            'iters_start': self.current_iters,
            'state_dict': self.model.state_dict(),
        }
        
        # Add optimizer state if available
        if hasattr(self, 'optimizer'):
            ckpt['optimizer'] = self.optimizer.state_dict()
        
        # Add AMP scaler state if available
        if self.amp_scaler is not None:
            ckpt['amp_scaler'] = self.amp_scaler.state_dict()
        
        # Add LR scheduler state if available
        if self.lr_scheduler is not None:
            ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        # Save checkpoint
        torch.save(ckpt, ckpt_path)
        print(f"✓ Checkpoint saved: {ckpt_path}")
        
        # Save EMA checkpoint separately if EMA is enabled
        if self.ema is not None:
            ema_ckpt_path = self.ckpt_dir / f'ema_model_{self.current_iters}.pth'
            torch.save(self.ema.state_dict(), ema_ckpt_path)
            print(f"✓ EMA checkpoint saved: {ema_ckpt_path}")
        
        return ckpt_path
    
    def resume_from_ckpt(self, ckpt_path):
        """
        Resume training from a checkpoint.
        
        Loads:
        - Model state dict
        - Optimizer state dict
        - AMP scaler state (if AMP is enabled)
        - LR scheduler state (if scheduler exists)
        - Current iteration number
        - Restores LR schedule by replaying adjust_lr for previous iterations
        
        Args:
            ckpt_path: Path to checkpoint file (.pth)
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        if not ckpt_path.endswith('.pth'):
            raise ValueError(f"Checkpoint file must have .pth extension: {ckpt_path}")
        
        print(f"\n{'=' * 100}")
        print(f"Resuming from checkpoint: {ckpt_path}")
        print(f"{'=' * 100}")
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # Load model state dict
        if 'state_dict' in ckpt:
            self.model.load_state_dict(ckpt['state_dict'])
            print(f"✓ Loaded model state dict")
        else:
            # If checkpoint is just the state dict
            self.model.load_state_dict(ckpt)
            print(f"✓ Loaded model state dict (direct)")
        
        # Load optimizer state dict (must be done after optimizer is created)
        if 'optimizer' in ckpt:
            if hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(ckpt['optimizer'])
                print(f"✓ Loaded optimizer state dict")
            else:
                print(f"⚠ Warning: Optimizer state found in checkpoint but optimizer not yet created.")
                print(f"  Optimizer will be loaded after setup_optimization() is called.")
                self._pending_optimizer_state = ckpt['optimizer']
        
        # Load AMP scaler state
        if 'amp_scaler' in ckpt:
            if hasattr(self, 'amp_scaler') and self.amp_scaler is not None:
                self.amp_scaler.load_state_dict(ckpt['amp_scaler'])
                print(f"✓ Loaded AMP scaler state")
            else:
                print(f"⚠ Warning: AMP scaler state found but AMP not enabled or scaler not created.")
        
        # Load LR scheduler state
        if 'lr_scheduler' in ckpt:
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                print(f"✓ Loaded LR scheduler state")
            else:
                print(f"⚠ Warning: LR scheduler state found but scheduler not yet created.")
                self._pending_lr_scheduler_state = ckpt['lr_scheduler']
        
        # Load EMA state if available (must be done after EMA is initialized)
        # EMA checkpoint naming: ema_model_{iters}.pth (matches save pattern)
        ckpt_path_obj = Path(ckpt_path)
        # Extract iteration number from checkpoint name (e.g., "model_10000.pth" -> "10000")
        if 'iters_start' in ckpt:
            iters = ckpt['iters_start']
            ema_ckpt_path = ckpt_path_obj.parent / f"ema_model_{iters}.pth"
        else:
            # Fallback: try to extract from filename
            try:
                iters = int(ckpt_path_obj.stem.split('_')[-1])
                ema_ckpt_path = ckpt_path_obj.parent / f"ema_model_{iters}.pth"
            except:
                ema_ckpt_path = None
        
        if ema_ckpt_path is not None and ema_ckpt_path.exists() and self.ema is not None:
            ema_ckpt = torch.load(ema_ckpt_path, map_location=self.device)
            self.ema.load_state_dict(ema_ckpt)
            print(f"✓ Loaded EMA state from: {ema_ckpt_path}")
        elif ema_ckpt_path is not None and ema_ckpt_path.exists() and self.ema is None:
            print(f"⚠ Warning: EMA checkpoint found but EMA not enabled. Skipping EMA load.")
        elif self.ema is not None:
            print(f"⚠ Warning: EMA enabled but no EMA checkpoint found. Starting with fresh EMA.")
        
        # Restore iteration number
        if 'iters_start' in ckpt:
            self.iters_start = ckpt['iters_start']
            self.current_iters = ckpt['iters_start']
            print(f"✓ Resuming from iteration: {self.iters_start}")
        else:
            print(f"⚠ Warning: No iteration number found in checkpoint. Starting from 0.")
            self.iters_start = 0
            self.current_iters = 0
        
        # Note: LR schedule restoration will be done after setup_optimization()
        # Store the iteration number for later restoration
        self._resume_iters = self.iters_start
        
        print(f"{'=' * 100}\n")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def log_step_train(self, loss, hr_latent, lr_latent, x_t, pred, phase='train'):
        """
        Component 8: Log training metrics and images to WandB.
        
        Logs:
        - Loss and learning rate (at log_freq[0] intervals)
        - Training images: HR, LR, and predictions (at log_freq[1] intervals)
        - Elapsed time for checkpoint intervals
        
        Args:
            loss: Training loss value (float)
            hr_latent: High-resolution latent tensor (B, C, 64, 64)
            lr_latent: Low-resolution latent tensor (B, C, 64, 64)
            x_t: Noisy input tensor (B, C, 64, 64)
            pred: Model prediction (x0_pred - clean HR latent) (B, C, 64, 64)
            phase: Training phase ('train' or 'val')
        """
        # Log loss and learning rate at log_freq[0] intervals
        if self.current_iters % log_freq[0] == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Get timing info if available (passed from training_step)
            timing_info = {}
            if hasattr(self, '_last_timing'):
                timing_info = {
                    'train/step_time': self._last_timing.get('step_time', 0),
                    'train/forward_time': self._last_timing.get('forward_time', 0),
                    'train/backward_time': self._last_timing.get('backward_time', 0),
                    'train/iterations_per_sec': 1.0 / self._last_timing.get('step_time', 1.0) if self._last_timing.get('step_time', 0) > 0 else 0
                }
            
            wandb.log({
                'loss': loss,
                'learning_rate': current_lr,
                'step': self.current_iters,
                **timing_info
            })
            
            # Print to console
            timing_str = ""
            if hasattr(self, '_last_timing') and self._last_timing.get('step_time', 0) > 0:
                timing_str = f", Step: {self._last_timing['step_time']:.3f}s, Forward: {self._last_timing['forward_time']:.3f}s, Backward: {self._last_timing['backward_time']:.3f}s"
            print(f"Train: {self.current_iters:06d}/{iterations:06d}, "
                  f"Loss: {loss:.6f}, LR: {current_lr:.2e}{timing_str}")
        
        # Log images at log_freq[1] intervals
        if self.current_iters % log_freq[1] == 0:
            with torch.no_grad():
                # Decode latents to pixel space for visualization
                # Take first sample from batch
                hr_pixel = self.autoencoder.decode(hr_latent[0:1])  # (1, 3, 256, 256)
                lr_pixel = self.autoencoder.decode(lr_latent[0:1])  # (1, 3, 256, 256)
                
                # Decode noisy input for visualization
                x_t_pixel = self.autoencoder.decode(x_t[0:1])  # (1, 3, 256, 256)
                
                # Decode predicted x0 (clean HR latent) for visualization
                pred_pixel = self.autoencoder.decode(pred[0:1])  # (1, 3, 256, 256)
                
                # Log images to WandB
                wandb.log({
                    f'{phase}/hr_sample': wandb.Image(hr_pixel[0].cpu().clamp(0, 1)),
                    f'{phase}/lr_sample': wandb.Image(lr_pixel[0].cpu().clamp(0, 1)),
                    f'{phase}/noisy_input': wandb.Image(x_t_pixel[0].cpu().clamp(0, 1)),
                    f'{phase}/pred_sample': wandb.Image(pred_pixel[0].cpu().clamp(0, 1)),
                    'step': self.current_iters
                })
        
        # Track elapsed time for checkpoint intervals
        if self.current_iters % save_freq == 1:
            self.tic = time.time()
        if self.current_iters % save_freq == 0 and self.tic is not None:
            self.toc = time.time()
            elapsed = self.toc - self.tic
            print(f"Elapsed time for {save_freq} iterations: {elapsed:.2f}s")
            print("=" * 100)
    
    def validation(self):
        """
        Run validation on validation dataset with full diffusion sampling loop.
        
        Performs iterative denoising from t = T-1 down to t = 0, matching the
        original ResShift implementation. This is slower but more accurate than
        single-step prediction.
        
        Computes:
        - PSNR, SSIM, and LPIPS metrics
        - Logs validation images to WandB
        """
        if 'val' not in self.dataloaders:
            print("No validation dataset available. Skipping validation.")
            return
        
        print("\n" + "=" * 100)
        print("Running Validation")
        print("=" * 100)
        
        val_start = time.time()
        
        # Use EMA model for validation if enabled
        if use_ema_val and self.ema is not None:
            # Create EMA model copy if it doesn't exist
            if self.ema_model is None:
                from copy import deepcopy
                self.ema_model = deepcopy(self.model)
            # Load EMA state into EMA model
            self.ema.apply_to_model(self.ema_model)
            self.ema_model.eval()
            val_model = self.ema_model
            print("Using EMA model for validation")
        else:
            self.model.eval()
            val_model = self.model
            if use_ema_val and self.ema is None:
                print("⚠ Warning: use_ema_val=True but EMA not enabled. Using regular model.")
        
        val_iter = iter(self.dataloaders['val'])
        
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_samples = 0
        
        total_sampling_time = 0.0
        total_forward_time = 0.0
        total_decode_time = 0.0
        total_metric_time = 0.0
        
        with torch.no_grad():
            for batch_idx, (hr_latent, lr_latent) in enumerate(val_iter):
                batch_start = time.time()
                # Move to device
                hr_latent = hr_latent.to(self.device)
                lr_latent = lr_latent.to(self.device)
                
                # Full diffusion sampling loop (iterative denoising)
                # Start from maximum timestep and iterate backwards: T-1 → T-2 → ... → 1 → 0
                sampling_start = time.time()
                
                # Initialize x_t at maximum timestep (T-1)
                # Start from LR with maximum noise
                residual = (lr_latent - hr_latent)
                epsilon_init = torch.randn_like(hr_latent)
                eta_max = self.eta[T - 1]
                x_t = hr_latent + eta_max * residual + k * torch.sqrt(eta_max) * epsilon_init
                
                # Track forward pass time during sampling
                sampling_forward_time = 0.0
                
                # Iterative sampling: denoise from t = T-1 down to t = 0
                for t_step in range(T - 1, -1, -1):  # T-1, T-2, ..., 1, 0
                    t = torch.full((hr_latent.shape[0],), t_step, device=self.device, dtype=torch.long)
                    
                    # Scale input for training stability (normalize variance across timesteps)
                    x_t_scaled = self._scale_input(x_t, t)
                    # Predict x0 from current noisy state x_t
                    forward_start = time.time()
                    x0_pred = val_model(x_t_scaled, t, lq=lr_latent)
                    sampling_forward_time += time.time() - forward_start
                    
                    # If not the last step, compute x_{t-1} from predicted x0
                    if t_step > 0:
                        # Compute x_{t-1} using forward formula with predicted x0
                        # x_{t-1} = x0_pred + eta_{t-1} * (lr - x0_pred) + k * sqrt(eta_{t-1}) * epsilon
                        # For deterministic sampling, we can use the predicted x0 directly
                        # and add scaled noise based on the schedule
                        eta_t_minus_1 = self.eta[t_step - 1]
                        
                        # Extract the noise component from current x_t
                        # epsilon = (x_t - x0_pred - eta_t * (lr - x0_pred)) / (k * sqrt(eta_t))
                        eta_t = self.eta[t_step]
                        epsilon_pred = (x_t - x0_pred - eta_t * (lr_latent - x0_pred)) / (k * torch.sqrt(eta_t) + 1e-8)
                        
                        # Compute x_{t-1} using predicted x0 and extracted noise
                        x_t = x0_pred + eta_t_minus_1 * (lr_latent - x0_pred) + k * torch.sqrt(eta_t_minus_1) * epsilon_pred
                    else:
                        # Final step: use predicted x0 as final output
                        x_t = x0_pred
                
                # Final prediction after full sampling loop
                x0_final = x_t
                
                sampling_time = time.time() - sampling_start
                total_sampling_time += sampling_time
                total_forward_time += sampling_forward_time
                
                # Decode latents to pixel space for metrics and visualization
                decode_start = time.time()
                hr_pixel = self.autoencoder.decode(hr_latent)
                lr_pixel = self.autoencoder.decode(lr_latent)
                sr_pixel = self.autoencoder.decode(x0_final)  # Final SR output after full sampling
                decode_time = time.time() - decode_start
                total_decode_time += decode_time
                
                # Convert to [0, 1] range if needed
                hr_pixel = hr_pixel.clamp(0, 1)
                sr_pixel = sr_pixel.clamp(0, 1)
                
                # Compute metrics using simple functions
                metric_start = time.time()
                batch_psnr = compute_psnr(hr_pixel, sr_pixel)
                total_psnr += batch_psnr * hr_latent.shape[0]
                
                batch_ssim = compute_ssim(hr_pixel, sr_pixel)
                total_ssim += batch_ssim * hr_latent.shape[0]
                
                batch_lpips = compute_lpips(hr_pixel, sr_pixel, self.lpips_model)
                total_lpips += batch_lpips * hr_latent.shape[0]
                metric_time = time.time() - metric_start
                total_metric_time += metric_time
                
                num_samples += hr_latent.shape[0]
                
                batch_time = time.time() - batch_start
                
                # Print timing for first batch
                if batch_idx == 0:
                    print(f"\nValidation Batch 0 Timing:")
                    print(f"  - Sampling loop: {sampling_time:.3f}s ({sampling_forward_time:.3f}s forward)")
                    print(f"  - Decoding: {decode_time:.3f}s")
                    print(f"  - Metrics: {metric_time:.3f}s")
                    print(f"  - Total batch: {batch_time:.3f}s")
                
                # Log validation images periodically
                if batch_idx == 0:
                    wandb.log({
                        'val/hr_sample': wandb.Image(hr_pixel[0].cpu()),
                        'val/lr_sample': wandb.Image(lr_pixel[0].cpu()),
                        'val/sr_sample': wandb.Image(sr_pixel[0].cpu()),
                        'step': self.current_iters
                    })
        
        # Compute average metrics and timing
        val_total_time = time.time() - val_start
        num_batches = batch_idx + 1
        
        if num_samples > 0:
            mean_psnr = total_psnr / num_samples
            mean_ssim = total_ssim / num_samples
            mean_lpips = total_lpips / num_samples
            
            avg_sampling_time = total_sampling_time / num_batches
            avg_forward_time = total_forward_time / num_batches
            avg_decode_time = total_decode_time / num_batches
            avg_metric_time = total_metric_time / num_batches
            avg_batch_time = val_total_time / num_batches
            
            print(f"\nValidation Metrics:")
            print(f"  - PSNR: {mean_psnr:.2f} dB")
            print(f"  - SSIM: {mean_ssim:.4f}")
            print(f"  - LPIPS: {mean_lpips:.4f}")
            
            print(f"\nValidation Timing (Total: {val_total_time:.2f}s, {num_batches} batches):")
            print(f"  - Avg sampling loop: {avg_sampling_time:.3f}s/batch ({avg_forward_time:.3f}s forward)")
            print(f"  - Avg decoding: {avg_decode_time:.3f}s/batch")
            print(f"  - Avg metrics: {avg_metric_time:.3f}s/batch")
            print(f"  - Avg batch time: {avg_batch_time:.3f}s/batch")
            
            wandb.log({
                'val/psnr': mean_psnr,
                'val/ssim': mean_ssim,
                'val/lpips': mean_lpips,
                'val/total_time': val_total_time,
                'val/avg_sampling_time': avg_sampling_time,
                'val/avg_forward_time': avg_forward_time,
                'val/avg_decode_time': avg_decode_time,
                'val/avg_metric_time': avg_metric_time,
                'val/avg_batch_time': avg_batch_time,
                'val/num_batches': num_batches,
                'val/num_samples': num_samples,
                'step': self.current_iters
            })
        
        print("=" * 100)
        
        # Set model back to training mode
        self.model.train()
        if self.ema_model is not None:
            self.ema_model.train()  # Keep in sync, but won't be used for training


# Note: Main training script is in train.py
# This file contains the Trainer class implementation
