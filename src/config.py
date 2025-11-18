"""
Configuration file with all training, model, and data parameters.
"""
import os
import torch
from pathlib import Path

# ============================================================================
# Project Settings
# ============================================================================
_project_root = Path(__file__).parent.parent

# ============================================================================
# Device Settings
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Training Parameters
# ============================================================================
# Learning rate
lr = 5e-5  # Original ResShift setting
lr_min = 2e-5
lr_schedule = None
learning_rate = lr  # Alias for backward compatibility
warmup_iterations = 100  # ~12.5% of total iterations (800), linear warmup from 0 to base_lr

# Dataloader
batch = [64, 64]  # Original ResShift: adjust based on your GPU memory
batch_size = batch[0]  # Use first value from batch list
microbatch = 100
num_workers = 4
prefetch_factor = 2

# Optimization settings
weight_decay = 0
ema_rate = 0.999
iterations = 800  # 64 epochs for DIV2K (800 images / 64 batch_size = 12.5 batches per epoch)

# Save logging
save_freq = 200
log_freq = [50, 100]  # [training loss, training images]
local_logging = True
tf_logging = False

# Validation settings
use_ema_val = True
val_freq = 100  # Run validation every 100 iterations
val_y_channel = True
val_resolution = 64  # model.params.lq_size
val_padding_mode = "reflect"

# Training setting
use_amp = True  # Mixed precision training
seed = 123456
global_seeding = False

# Model compile
compile_flag = True
compile_mode = "reduce-overhead"

# ============================================================================
# Diffusion/Noise Schedule Parameters
# ============================================================================
sf = 4
schedule_name = "exponential"
schedule_power = 0.3  # Original ResShift setting
etas_end = 0.99  # Original ResShift setting
T = 15  # Original ResShift: 15 timesteps
min_noise_level = 0.04  # Original ResShift setting
eta_1 = min_noise_level  # Alias for backward compatibility
eta_T = etas_end  # Alias for backward compatibility
p = schedule_power  # Alias for backward compatibility
kappa = 2.0
k = kappa  # Alias for backward compatibility
weighted_mse = False
predict_type = "xstart"  # Predict x0, not noise (key difference!)
timestep_respacing = None
scale_factor = 1.0
normalize_input = True
latent_flag = True  # Working in latent space

# ============================================================================
# Model Architecture Parameters
# ============================================================================
# ResShift model architecture based on model_channels and channel_mult
# Initial Conv: 3 → 160
# Encoder Stage 1: 160 → 320 (downsample to 128x128)
# Encoder Stage 2: 320 → 320 (downsample to 64x64)
# Encoder Stage 3: 320 → 640 (downsample to 32x32)
# Encoder Stage 4: 640 (no downsampling, stays 32x32)
# Decoder Stage 1: 640 → 320 (upsample to 64x64)
# Decoder Stage 2: 320 → 320 (upsample to 128x128)
# Decoder Stage 3: 320 → 160 (upsample to 256x256)
# Decoder Stage 4: 160 → 3 (final output)

# Model params from ResShift configuration
image_size = 64  # Latent space: 64×64 (not 256×256 pixel space)
in_channels = 3
model_channels = 160  # Original ResShift: base channels
out_channels = 3
attention_resolutions = [64, 32, 16, 8]  # Latent space resolutions
dropout = 0
channel_mult = [1, 2, 2, 4]  # Original ResShift: 160, 320, 320, 640 channels
num_res_blocks = [2, 2, 2, 2]
conv_resample = True
dims = 2
use_fp16 = False
num_head_channels = 32
use_scale_shift_norm = True
resblock_updown = False
swin_depth = 2
swin_embed_dim = 192  # Original ResShift setting
window_size = 8  # Original ResShift setting (not 7)
mlp_ratio = 2.0  # Original ResShift uses 2.0, not 4
cond_lq = True  # Enable LR conditioning
lq_size = 64  # LR latent size (same as image_size)

# U-Net architecture parameters based on ResShift configuration
# Initial conv: 3 → model_channels * channel_mult[0] = 160
initial_conv_out_channels = model_channels * channel_mult[0]  # 160

# Encoder stage channels (based on channel_mult progression)
es1_in_channels = initial_conv_out_channels  # 160
es1_out_channels = model_channels * channel_mult[1]  # 320
es2_in_channels = es1_out_channels  # 320
es2_out_channels = model_channels * channel_mult[2]  # 320
es3_in_channels = es2_out_channels  # 320
es3_out_channels = model_channels * channel_mult[3]  # 640
es4_in_channels = es3_out_channels  # 640
es4_out_channels = es3_out_channels  # 640 (no downsampling)

# Decoder stage channels (reverse of encoder)
ds1_in_channels = es4_out_channels  # 640
ds1_out_channels = es2_out_channels  # 320
ds2_in_channels = ds1_out_channels  # 320
ds2_out_channels = es2_out_channels  # 320
ds3_in_channels = ds2_out_channels  # 320
ds3_out_channels = es1_out_channels  # 160
ds4_in_channels = ds3_out_channels  # 160
ds4_out_channels = initial_conv_out_channels  # 160

# Other model parameters
n_groupnorm_groups = 8  # Standard value
shift_size = window_size // 2  # Shift size for shifted window attention (should be window_size // 2, not swin_depth)
timestep_embed_dim = model_channels * 4  # Original ResShift: 160 * 4 = 640
num_heads = num_head_channels  # Note: config has num_head_channels, but we need num_heads

# ============================================================================
# Autoencoder Parameters (from YAML, for reference)
# ============================================================================
autoencoder_ckpt_path = "pretrained_weights/autoencoder_vq_f4.pth"
autoencoder_use_fp16 = False  # Temporarily disabled for CPU testing (FP16 is slow/hangs on CPU)
autoencoder_embed_dim = 3
autoencoder_n_embed = 8192
autoencoder_double_z = False
autoencoder_z_channels = 3
autoencoder_resolution = 256
autoencoder_in_channels = 3
autoencoder_out_ch = 3
autoencoder_ch = 128
autoencoder_ch_mult = [1, 2, 4]
autoencoder_num_res_blocks = 2
autoencoder_attn_resolutions = []
autoencoder_dropout = 0.0
autoencoder_padding_mode = "zeros"

# ============================================================================
# Degradation Parameters (used by realesrgan.py)
# ============================================================================
# Blur kernel settings (used for both first and second degradation)
blur_kernel_size = 21
kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]

# First degradation stage
resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
resize_range = [0.15, 1.5]
gaussian_noise_prob = 0.5
noise_range = [1, 30]
poisson_scale_range = [0.05, 3.0]
gray_noise_prob = 0.4
jpeg_range = [30, 95]
data_train_blur_sigma = [0.2, 3.0]
data_train_betag_range = [0.5, 4.0]
data_train_betap_range = [1, 2.0]
data_train_sinc_prob = 0.1

# Second degradation stage
second_order_prob = 0.5
second_blur_prob = 0.8
resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
resize_range2 = [0.3, 1.2]
gaussian_noise_prob2 = 0.5
noise_range2 = [1, 25]
poisson_scale_range2 = [0.05, 2.5]
gray_noise_prob2 = 0.4
jpeg_range2 = [30, 95]
data_train_blur_kernel_size2 = 15
data_train_blur_sigma2 = [0.2, 1.5]
data_train_betag_range2 = [0.5, 4.0]
data_train_betap_range2 = [1, 2.0]
data_train_sinc_prob2 = 0.1

# Final sinc filter
data_train_final_sinc_prob = 0.8
final_sinc_prob = data_train_final_sinc_prob  # Alias for backward compatibility

# Other degradation settings
gt_size = 256
resize_back = False
use_sharp = False

# ============================================================================
# Data Parameters
# ============================================================================
# Data paths - using defaults based on project structure
dir_HR = str(_project_root / "data" / "DIV2K_train_HR")
dir_LR = str(_project_root / "data" / "DIV2K_train_LR_bicubic" / "X4")
dir_valid_HR = str(_project_root / "data" / "DIV2K_valid_HR")
dir_valid_LR = str(_project_root / "data" / "DIV2K_valid_LR_bicubic" / "X4")

# Patch size (used by dataset)
patch_size = gt_size  # 256

# Scale factor (from degradation.sf)
scale = sf  # 4
