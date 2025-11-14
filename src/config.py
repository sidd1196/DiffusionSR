import torch 
import torch.nn as nn
import os

# Get project root directory (parent of src/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## data
scale = 4
patch_size = 256

# Dataset paths
# Note: os.path.join ignores previous args if given an absolute path
# So we use the absolute path directly
dir_HR = '/Users/akhilshekkari/Library/CloudStorage/GoogleDrive-shekkari@umd.edu/My Drive/train_HR_patches_256'
dir_LR = '/Users/akhilshekkari/Library/CloudStorage/GoogleDrive-shekkari@umd.edu/My Drive/train_LR_patches_256'

# Validation paths
dir_valid_HR = '/Users/akhilshekkari/Library/CloudStorage/GoogleDrive-shekkari@umd.edu/My Drive/valid_HR_patches_256'
dir_valid_LR = '/Users/akhilshekkari/Library/CloudStorage/GoogleDrive-shekkari@umd.edu/My Drive/valid_LR_patches_256'

# noise control params
T = 15
eta_1 = 0.001
eta_T = 0.999
p = 0.8

# es = encoder stage
es1_in_channels  = 64
es1_out_channels = 64
es2_in_channels  = 128
es2_out_channels = 128
es3_in_channels  = 256
es3_out_channels = 256
es4_in_channels  = 512
es4_out_channels = 512

# swin parameters
shift_size = 3
window_size = 7
num_heads = 8

# bottleneck
bn_in_channels = 512
bn_out_channels = 512

# ds = decoder stage
ds1_in_channels = 512
ds1_out_channels = 256
ds2_in_channels = 256
ds2_out_channels = 128
ds3_in_channels = 128
ds3_out_channels = 64
ds4_in_channels = 64
ds4_out_channels = 64

## general
timestep_embed_dim = 128
n_groupnorm_groups = 8
initial_conv_out_channels = 64

## training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
