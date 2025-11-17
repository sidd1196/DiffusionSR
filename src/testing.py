import torch
import torch.nn as nn
from model import FullUNET
from noiseControl import resshift_schedule
from torch.utils.data import DataLoader
from data import mini_dataset, train_dataset, get_vqgan_model
import torch.optim as optim
from config import (batch_size, device, learning_rate, iterations, 
                    weight_decay, T, k, _project_root)
import wandb
import os
from dotenv import load_dotenv

# Load environment variables from .env file (looks for .env in project root)
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
        "optimizer": "Adam",
        "betas": (0.9, 0.999),
        "grad_clip": 1.0,
        "criterion": "MSE",
        "device": str(device),
        "training_space": "latent_64x64"
    }
)

# Load VQGAN for decoding latents for visualization
vqgan = get_vqgan_model()

train_dl = DataLoader(mini_dataset, batch_size=batch_size, shuffle=True)

# Get a batch - now returns 64x64 latents
hr_latent, lr_latent = next(iter(train_dl))

hr_latent = hr_latent.to(device)  # (B, C, 64, 64) - HR latent
lr_latent = lr_latent.to(device)  # (B, C, 64, 64) - LR latent

eta = resshift_schedule().to(device)
eta = eta[:, None, None, None]   # shape (T,1,1,1)
residual = (lr_latent - hr_latent)  # Residual in latent space
model = FullUNET()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
steps = iterations

# Watch model for gradients/parameters
wandb.watch(model, log="all", log_freq=10)
for step in range(steps):
    model.train()
    # take random timestep (0 to T-1)
    t = torch.randint(0, T, (batch_size,)).to(device)

    # add the noise in latent space
    epsilon = torch.randn_like(hr_latent)  # Noise in latent space
    eta_t = eta[t]
    x_t = hr_latent + eta_t * residual + k * torch.sqrt(eta_t) * epsilon
    # send the same patch in model forwardpass across different timestamps per each step
    # lr_latent is the low-resolution latent used for conditioning
    pred = model(x_t, t, lq=lr_latent)
    optimizer.zero_grad()
    loss = criterion(pred, epsilon)
    wandb.log({
    "loss": loss.item(),
    "step": step,
    "learning_rate": optimizer.param_groups[0]['lr']
    })
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 50 == 0:
        # Decode latents to pixel space for visualization
        with torch.no_grad():
            hr_pixel = vqgan.decode(hr_latent[0:1])  # (1, 3, 256, 256)
            lr_pixel = vqgan.decode(lr_latent[0:1])  # (1, 3, 256, 256)
            pred_pixel = vqgan.decode(x_t[0:1])  # (1, 3, 256, 256)
        
        wandb.log({
            "hr_sample": wandb.Image(hr_pixel[0].cpu().clamp(0, 1)),
            "lr_sample": wandb.Image(lr_pixel[0].cpu().clamp(0, 1)),
            "pred_sample": wandb.Image(pred_pixel[0].cpu().clamp(0, 1))
        })
    print(f'loss at step {step + 1} is {loss}')

wandb.finish()
