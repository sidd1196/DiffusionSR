import torch
import torch.nn as nn
from model import FullUNET
from noiseControl import resshift_schedule
from torch.utils.data import DataLoader
from data import mini_dataset, train_dataset
import torch.optim as optim
from config import batch_size, device
import wandb
import os
from dotenv import load_dotenv

# Load environment variables from .env file (looks for .env in project root)
from config import _project_root
load_dotenv(os.path.join(_project_root, '.env'))

wandb.init(
    project="diffusionsr",
    name="reshift_training",
    config={
        "learning_rate": 1e-4,
        "batch_size": batch_size,
        "steps": 150,
        "model": "ResShift",
        "T": 15,
        "k": 1,
        "optimizer": "Adam",
        "betas": (0.9, 0.999),
        "grad_clip": 1.0,
        "criterion": "MSE",
        "device": str(device)
    }
)
train_dl = DataLoader(mini_dataset, batch_size=batch_size, shuffle=True)

hr, lr = next(iter(train_dl))

hr = hr.to(device)
lr = lr.to(device)
k = 1
eta = resshift_schedule().to(device)
eta = eta[:, None, None, None]   # shape (15,1,1,1)
residual = (lr - hr)
model = FullUNET()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
steps = 150

# Watch model for gradients/parameters
wandb.watch(model, log="all", log_freq=10)
for step in range(steps):
    model.train()
    # take random timestep
    t = torch.randint(0, 14, (batch_size,)).to(device)

    # add the noise
    epsilon = torch.randn_like(hr)
    eta_t = eta[t]
    x_t = hr + eta_t * residual + k * torch.sqrt(eta_t) * epsilon
    # send the same patch in model forwardpass across different timestamps per each step
    pred = model(x_t, t)
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
        wandb.log({
            "hr_sample": wandb.Image(hr[0]),
            "lr_sample": wandb.Image(lr[0]),
            "pred_sample": wandb.Image(pred[0])
        })
    print(f'loss at step {step + 1} is {loss}')

wandb.finish()
