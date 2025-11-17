"""
VQGAN Autoencoder module for encoding/decoding images to/from latent space.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Handle import of ldm from latent-diffusion repository
# Check if ldm directory exists locally (from latent-diffusion repo)
_ldm_path = Path(__file__).parent.parent / "ldm"
if _ldm_path.exists() and str(_ldm_path) not in sys.path:
    sys.path.insert(0, str(_ldm_path.parent))

try:
    from ldm.models.autoencoder import VQModelTorch
except ImportError:
    # Fallback: try importing from site-packages if latent-diffusion is installed
    try:
        import importlib.util
        spec = importlib.util.find_spec("ldm.models.autoencoder")
        if spec is None:
            raise ImportError("Could not find ldm.models.autoencoder")
        from ldm.models.autoencoder import VQModelTorch
    except ImportError as e:
        raise ImportError(
            "Could not import VQModelTorch from ldm.models.autoencoder. "
            "Please ensure the latent-diffusion repository is cloned and the ldm directory exists, "
            "or install latent-diffusion package. Error: " + str(e)
        )
from config import (
    autoencoder_ckpt_path, 
    autoencoder_use_fp16,
    autoencoder_embed_dim,
    autoencoder_n_embed,
    autoencoder_double_z,
    autoencoder_z_channels,
    autoencoder_resolution,
    autoencoder_in_channels,
    autoencoder_out_ch,
    autoencoder_ch,
    autoencoder_ch_mult,
    autoencoder_num_res_blocks,
    autoencoder_attn_resolutions,
    autoencoder_dropout,
    autoencoder_padding_mode,
    _project_root,
    device
)


def load_vqgan(ckpt_path=None, device=device):
    """
    Load VQGAN autoencoder from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file. If None, uses config path.
        device: Device to load model on.
    
    Returns:
        VQGAN model in eval mode.
    """
    if ckpt_path is None:
        ckpt_path = autoencoder_ckpt_path
    
    # Resolve path relative to project root
    if not Path(ckpt_path).is_absolute():
        ckpt_path = _project_root / ckpt_path
    
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"VQGAN checkpoint not found at: {ckpt_path}")
    
    print(f"Loading VQGAN from: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract state_dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
    
    # Create model architecture
    ddconfig = {
        'double_z': autoencoder_double_z,
        'z_channels': autoencoder_z_channels,
        'resolution': autoencoder_resolution,
        'in_channels': autoencoder_in_channels,
        'out_ch': autoencoder_out_ch,
        'ch': autoencoder_ch,
        'ch_mult': autoencoder_ch_mult,
        'num_res_blocks': autoencoder_num_res_blocks,
        'attn_resolutions': autoencoder_attn_resolutions,
        'dropout': autoencoder_dropout,
        'padding_mode': autoencoder_padding_mode,
    }
    
    model = VQModelTorch(
        ddconfig=ddconfig,
        n_embed=autoencoder_n_embed,
        embed_dim=autoencoder_embed_dim,
    )
    
    # Load state_dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    if autoencoder_use_fp16:
        model = model.half()
    
    print(f"VQGAN loaded successfully on {device}")
    return model


class VQGANWrapper(nn.Module):
    """
    Simple wrapper for VQGAN autoencoder.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def encode(self, x):
        """
        Encode image to latent space.
        
        Args:
            x: (B, 3, H, W) Image tensor in range [0, 1]
        
        Returns:
            z: (B, 3, H//4, W//4) Latent tensor
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        with torch.no_grad():
            # Normalize to [-1, 1] if needed
            if x.max() <= 1.0:
                x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            
            # Match model dtype (handle fp16 models)
            model_dtype = next(self.model.parameters()).dtype
            if x.dtype != model_dtype:
                x = x.to(model_dtype)
            
            # Ensure input is on same device as model
            model_device = next(self.model.parameters()).device
            if x.device != model_device:
                x = x.to(model_device)
            
            # Encode
            z = self.model.encode(x)
            
            # Extract latent from tuple/dict if needed
            if isinstance(z, (tuple, list)):
                z = z[0]
            elif isinstance(z, dict):
                z = z.get('z', z.get('latent', z))
            
            # Convert back to float32 for consistency
            if z.dtype != torch.float32:
                z = z.float()
            
            return z
    
    def decode(self, z):
        """
        Decode latent to image space.
        
        Args:
            z: (B, 3, H, W) Latent tensor
        
        Returns:
            x: (B, 3, H*4, W*4) Image tensor in range [0, 1]
        """
        with torch.no_grad():
            # Match model dtype (handle fp16 models)
            model_dtype = next(self.model.parameters()).dtype
            if z.dtype != model_dtype:
                z = z.to(model_dtype)
            
            # Decode
            x = self.model.decode(z)
            
            # Convert back to float32
            if x.dtype != torch.float32:
                x = x.float()
            
            # Normalize back to [0, 1]
            if x.min() < 0:
                x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            
            x = torch.clamp(x, 0, 1)
            return x


# Convenience function
def get_vqgan(ckpt_path=None, device=device):
    """Get VQGAN model instance."""
    model = load_vqgan(ckpt_path=ckpt_path, device=device)
    return VQGANWrapper(model)
