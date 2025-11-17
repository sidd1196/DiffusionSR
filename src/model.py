
import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
from config import (ds1_in_channels, ds1_out_channels, ds2_in_channels, ds2_out_channels, 
                    ds3_in_channels, ds3_out_channels, ds4_in_channels, ds4_out_channels, 
                    es1_in_channels, es1_out_channels, es2_in_channels, 
                    es2_out_channels, es3_in_channels, es3_out_channels, 
                    es4_in_channels, es4_out_channels, n_groupnorm_groups, shift_size,
                    timestep_embed_dim, initial_conv_out_channels, num_heads, window_size,
                    in_channels, dropout, mlp_ratio, swin_embed_dim, use_scale_shift_norm,
                    attention_resolutions, image_size)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def sinusoidal_embedding(timesteps, dim=timestep_embed_dim):
    """
    timesteps: (B,) int64 tensor
    dim: embedding dimension
    returns: (B, dim) tensor

    Just like how positional encodings are there in Transformers
    """
    device = timesteps.device
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    args = timesteps[:, None] * freq[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

class TimeEmbeddingMLP(nn.Module):
    def __init__(self, emb_dim, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, t_emb):
        return self.mlp(t_emb)   # (B, out_channels)

class InitialConv(nn.Module):
    '''
    Input :  We get input image concatenated with LR image (6 channels total)
    Output:  We send it to Encoder stage 1
    '''
    def __init__(self, input_channels=None):
        '''
        Input Shape --> [256 x 256 x input_channels]
        Output Shape --> [256 x 256 x initial_conv_out_channels]
        '''
        super().__init__()
        if input_channels is None:
            input_channels = in_channels
        self.net = nn.Conv2d(in_channels=input_channels, out_channels=initial_conv_out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    '''
    Inside the Residual block, channels remain same
    Input  : From previous Encoder stage / Initial Conv
    Output : Downsampling block and we save skip connection for correspoding decoder stage
    '''
    def __init__(self, in_channels, out_channels, sin_embed_dim = timestep_embed_dim, dropout_rate=dropout, use_scale_shift=use_scale_shift_norm):
        '''
        This ResBlock will be used by following inchannels [64, 128, 256, 512]
        This ResBlock will be used by following outchannels [64, 128, 256, 512]
        '''
        super().__init__()
        self.use_scale_shift = use_scale_shift

        ## 1st res block (in_layers)
        self.norm1 = nn.GroupNorm(num_groups = n_groupnorm_groups, num_channels = in_channels) ## num_groups 8 are standard it seems
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride=1, padding=1)

        ## timestamp embedding MLP
        # If use_scale_shift_norm, output 2*out_channels (for scale and shift)
        # Otherwise, output out_channels (for additive)
        embed_out_dim = 2 * out_channels if use_scale_shift else out_channels
        self.MLP_embed = TimeEmbeddingMLP(sin_embed_dim, out_channels=embed_out_dim)

        ## 2nd res block (out_layers)
        self.norm2 = nn.GroupNorm(num_groups = n_groupnorm_groups, num_channels = out_channels) ## num_groups 8 are standard it seems
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=3, stride=1, padding=1)

        ## skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb): ## t_emb is pre-computed time embedding (B, timestep_embed_dim)
        # in_layers: norm -> SiLU -> conv
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Time embedding conditioning
        emb_out = self.MLP_embed(t_emb)  # (B, embed_out_dim)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None, None]  # (B, embed_out_dim, 1, 1)
        
        if self.use_scale_shift:
            # FiLM conditioning: h = norm(h) * (1 + scale) + shift
            scale, shift = torch.chunk(emb_out, 2, dim=1)  # Each (B, out_channels, 1, 1)
            h = self.norm2(h) * (1 + scale) + shift
            h = self.act2(h)
            h = self.dropout(h)
            h = self.conv2(h)
        else:
            # Additive conditioning: h = h + emb_out
            h = h + emb_out
            h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        
        return h + self.skip(x)

    
class Downsample(nn.Module):
    '''
    A downsampling layer using strided convolution.
    Reduces spatial resolution by half (stride=2) while keeping channels the same.
    
    Note: Channel changes happen in ResBlocks, not in this downsample layer.
    This matches the original ResShift implementation when conv_resample=True.
    
    Input: From each encoder stage
    Output: To next encoder stage (same channels, half resolution)
    '''
    def __init__(self, in_channels, out_channels):
        '''
        Args:
            in_channels: Input channel count
            out_channels: Output channel count (should equal in_channels in our usage)
        '''
        super().__init__()
        # Strided convolution: 3x3 conv with stride=2, padding=1
        # This halves the spatial resolution (e.g., 64x64 -> 32x32)
        self.net = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.net(x)

class EncoderStage(nn.Module):
    '''
    Combine ResBlock and downsample here
    x --> resolution
    y --> channels
    Input:  [y, x, x]
    Output: [2y, x/2, x/2]
    '''
    def __init__(self, in_channels, out_channels, downsample = True, resolution=None, use_attention=False):
        super().__init__()
        self.res1 = ResidualBlock(in_channels = in_channels, out_channels = out_channels)
        # Add attention after first res block if resolution matches and use_attention is True
        self.attention = None
        if use_attention and resolution in attention_resolutions:
            # Create BasicLayer equivalent: 2 SwinTransformerBlocks (one with shift=0, one with shift=window_size//2)
            self.attention = nn.Sequential(
                SwinTransformerBlock(in_channels=out_channels, num_heads=num_heads, shift_size=0, 
                                   embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio),
                SwinTransformerBlock(in_channels=out_channels, num_heads=num_heads, shift_size=window_size // 2, 
                                   embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio)
            )
        self.res2 = ResidualBlock(in_channels = out_channels, out_channels = out_channels)
        # handling this for the last part of the encoder stage 4
        # Downsample only reduces spatial resolution, keeps channels the same
        # Channel changes happen in ResBlocks, not in downsample
        self.do_downsample = Downsample(out_channels, out_channels) if downsample else nn.Identity()
        self.downsample = self.do_downsample

    def forward(self, x, t_emb):
        out = self.res1(x, t_emb)   ## here out is h + skip(x)
        # Apply attention if present (attention doesn't use t_emb)
        if self.attention is not None:
            out = self.attention(out)
        out_skipconnection = self.res2(out, t_emb)
       # print(f'The shape after Encoder Stage before downsampling is {out.squeeze(dim = 0).shape}')
        out_downsampled = self.downsample(out_skipconnection)
       # print(f'The shape after Encoder Stage after downsampling is {out.squeeze(dim = 0).shape}')
        return out_downsampled, out_skipconnection

class FullEncoderModule(nn.Module):
    '''
    connect all 4 encoder stages(for now)

    '''
    def __init__(self, input_channels=None):
        '''
        Passing through Encoder stages 1 by 1
        Args:
            input_channels: Number of input channels (default: in_channels from config)
        '''
        super().__init__()
        if input_channels is None:
            input_channels = in_channels
        self.initial_conv = InitialConv(input_channels=input_channels)
        # Add attention after initial conv if 64x64 is in attention_resolutions
        self.attention_initial = None
        if image_size in attention_resolutions:
            self.attention_initial = nn.Sequential(
                SwinTransformerBlock(in_channels=initial_conv_out_channels, num_heads=num_heads, shift_size=0, 
                                   embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio),
                SwinTransformerBlock(in_channels=initial_conv_out_channels, num_heads=num_heads, shift_size=window_size // 2, 
                                   embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio)
            )
        # Track resolutions: after initial_conv=64, after stage1=32, after stage2=16, after stage3=8, after stage4=8
        self.encoderstage_1 = EncoderStage(es1_in_channels, es1_out_channels, downsample=True, resolution=image_size, use_attention=True)
        self.encoderstage_2 = EncoderStage(es2_in_channels, es2_out_channels, downsample=True, resolution=image_size // 2, use_attention=True)
        self.encoderstage_3 = EncoderStage(es3_in_channels, es3_out_channels, downsample=True, resolution=image_size // 4, use_attention=True)
        self.encoderstage_4 = EncoderStage(es4_in_channels, es4_out_channels, downsample=False, resolution=image_size // 8, use_attention=True)

    def forward(self, x, t_emb):
        out = self.initial_conv(x)
        # Apply attention after initial conv if present
        if self.attention_initial is not None:
            out = self.attention_initial(out)
        out_1, skip_1 = self.encoderstage_1(out, t_emb)
        #print(f'The shape after Encoder Stage 1 after downsampling is {out_1.shape}')
        out_2, skip_2 = self.encoderstage_2(out_1, t_emb)
        #print(f'The shape after Encoder Stage 2 after downsampling is {out_2.shape}')
        out_3, skip_3 = self.encoderstage_3(out_2, t_emb)
        #print(f'The shape after Encoder Stage 3 after downsampling is {out_3.shape}')
        out_4, skip_4 = self.encoderstage_4(out_3, t_emb)
        #print(f'The shape after Encoder Stage 4  is {out_4.shape}')
        # i think we should return these for correspoding decoder stages
        return (out_1, skip_1), (out_2, skip_2), (out_3, skip_3), (out_4, skip_4)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    Supports both shifted and non-shifted windows.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, (tuple, list)) else (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_ x H x N x C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0).to(attn.dtype)

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads = num_heads, shift_size=0, embed_dim=None, mlp_ratio_val=None):
        '''

        As soon as the input image comes (512 x 32 x 32), we divide this into
        16 patches of 512 x 7 x 7

        Each patch is then flattented and it becomes (49 x 512)
        Now think of this as 49 tokens having 512 embedding dim vector. Usually a feature map is representation of pixel in embedding.
        If we say 3 x 4 x 4, that means each pixel is represented in 3 dim vector. Here, 49 pixels/tokens are represented in 512 dim.
        we will have an embedding layer for this.

        '''
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads # Store num_heads for mask generation
        
        # Use embed_dim from config if provided, otherwise use in_channels
        self.embed_dim = embed_dim if embed_dim is not None else swin_embed_dim
        self.mlp_ratio = mlp_ratio_val if mlp_ratio_val is not None else mlp_ratio
        
        # Projection layers if embed_dim differs from in_channels
        if self.embed_dim != in_channels:
            self.proj_in = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
            self.proj_out = nn.Conv2d(self.embed_dim, in_channels, kernel_size=1)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
        
        # Use custom WindowAttention with relative position bias
        self.attn = WindowAttention(
            dim=self.embed_dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * self.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(self.embed_dim * self.mlp_ratio), self.embed_dim)
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # Attention mask for shifted windows
        if self.shift_size > 0:
            # Will be computed in forward based on input size
            self.register_buffer("attn_mask", None, persistent=False)
        else:
            self.attn_mask = None

    def get_windowed_tokens(self, x):
        '''
        In a window, how many pixels/tokens are there and what is its representation in terms of vec
        '''
        B, C, H, W = x.size()
        ws = self.window_size
        # move channel to last dim to make reshaping intuitive
        x = x.permute(0, 2, 3, 1).contiguous()        # (B, H, W, C)

        # reshape into blocks: (B, H//ws, ws, W//ws, ws, C)
        x = x.view(B, H // ws, ws, W // ws, ws, C)

         # reorder to (B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, Nh, Nw, ws, ws, C)

        # merge windows: (B * Nh * Nw, ws * ws, C)
        windows_tokens = x.view(-1, ws * ws, C)

        return windows_tokens

    def window_reverse(self, windows, H, W, B):
        """Merge windows back to feature map."""
        ws = self.window_size
        num_windows_h = H // ws
        num_windows_w = W // ws
        x = windows.view(B, num_windows_h, num_windows_w, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    def calculate_mask(self, H, W, device):
        """Calculate attention mask for SW-MSA."""
        if self.shift_size == 0:
            return None
        
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Convert to (B, C, H, W) format for window_partition
        img_mask = img_mask.permute(0, 3, 1, 2).contiguous()  # (1, 1, H, W)
        mask_windows = self.get_windowed_tokens(img_mask)  # (num_windows, ws*ws, 1)
        mask_windows = mask_windows.squeeze(-1)  # (num_windows, ws*ws)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # shape: (num_windows, ws*ws, ws*ws)

        return attn_mask

    def forward(self, x):
        # pad the input first(since we are using 7x7 window, we gotta make our image from 32x32 to 35x35)
        '''
        Here there are two types of swin blocks.
        1. Windowed swin block
        2. shifted windowed swin block

        In our code we use both these blocks one after the other. The difference is the first computes local attention, without shifting.
        The second, shifts first, them computes local attention, then shifts it back.
        '''
        B, C, H, W = x.size()
        
        # Project to embed_dim if needed
        x = self.proj_in(x)  # (B, embed_dim, H, W)
        C_emb = x.shape[1]
        
        # Save shortcut AFTER projection (in embed_dim space for residual)
        shortcut = x
        
        # Pad if needed
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            shortcut = F.pad(shortcut, (0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[2], x.shape[3]

        # Normalize BEFORE windowing (original behavior)
        # Convert to (B, H, W, C) for LayerNorm
        x_norm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_norm = self.norm1(x_norm)  # Normalize spatial features
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x_norm

        # Partition windows
        x_windows = self.get_windowed_tokens(shifted_x)  # (num_windows*B, ws*ws, C)

        # Calculate mask for shifted windows
        if self.shift_size > 0:
            mask = self.calculate_mask(H_pad, W_pad, x.device)
        else:
            mask = None

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=mask)  # (num_windows*B, ws*ws, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C_emb)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad, B)  # (B, C, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        # Crop padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H_pad, :W_pad]
            shortcut = shortcut[:, :, :H_pad, :W_pad]

        # Residual connection around attention (original: shortcut + drop_path(x))
        x = shortcut + x  # Add in embed_dim space

        # FFN
        # Convert to (B, H, W, C) for LayerNorm
        x_norm2 = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_norm2 = self.norm2(x_norm2)  # (B, H, W, C)
        x_mlp = self.mlp(x_norm2)  # (B, H, W, C)
        x_mlp = x_mlp.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        # Residual connection around MLP
        x = x + x_mlp

        # Project back to in_channels if needed
        if self.embed_dim != C:
            x = self.proj_out(x)  # (B, in_channels, H, W)

        # Crop to original size
        x = x[:, :, :H, :W]

        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels = es4_out_channels, out_channels = ds1_in_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels = in_channels, out_channels = out_channels)
        # Use swin_embed_dim from config for projection
        self.swintransformer1 = SwinTransformerBlock(in_channels = out_channels, num_heads = num_heads, shift_size=0, embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio)
        self.swintransformer2 = SwinTransformerBlock(in_channels = out_channels, num_heads = num_heads, shift_size=window_size // 2, embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio)
        self.res2 = ResidualBlock(in_channels = out_channels, out_channels = out_channels)

    def forward(self, x, t_emb):
        res_out = self.res1(x, t_emb)
        swin_out_1 = self.swintransformer1(res_out)
       # print(f'swin_out_1 shape is {swin_out_1.shape}')
        swin_out_2 = self.swintransformer2(swin_out_1)
       # print(f'swin_out_2 shape is {swin_out_2.shape}')
        res_out_2 = self.res2(swin_out_2, t_emb)
        return res_out_2


class Upsample(nn.Module):
    '''
    Just increases resolution
    Input: From each decoder stage
    Output: To next decoder stage
    '''
    def __init__(self, in_channels, out_channels):
        '''
        Our target is to half the resolution and double the channels
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
    def forward(self, x):
        return self.net(x)


class DecoderStage(nn.Module):
    """
    Decoder block:
      - Optional upsample
      - Concatenate skip (channel dimension doubles)
      - Two residual blocks
    """
    def __init__(self, in_channels, skip_channels, out_channels, upsample=True, resolution=None, use_attention=False):
        super().__init__()

        # Upsample first, but keep same number of channels
        self.upsample = Upsample(in_channels, in_channels) if upsample else nn.Identity()

        #
        merged_channels = in_channels + skip_channels

        # First ResBlock processes merged tensor
        self.res1 = ResidualBlock(in_channels = merged_channels, out_channels=out_channels)
        
        # Add attention after first res block if resolution matches and use_attention is True
        self.attention = None
        if use_attention and resolution in attention_resolutions:
            # Create BasicLayer equivalent: 2 SwinTransformerBlocks (one with shift=0, one with shift=window_size//2)
            self.attention = nn.Sequential(
                SwinTransformerBlock(in_channels=out_channels, num_heads=num_heads, shift_size=0, 
                                   embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio),
                SwinTransformerBlock(in_channels=out_channels, num_heads=num_heads, shift_size=window_size // 2, 
                                   embed_dim=swin_embed_dim, mlp_ratio_val=mlp_ratio)
            )

        # Second ResBlock keeps output channels the same
        self.res2 = ResidualBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, skip, t_emb):
        """
        x    : (B, C, H, W) decoder input
        skip : (B, C_skip, H, W) encoder skip feature
        t_emb: (B, timestep_embed_dim) pre-computed time embedding
        """
        x = self.upsample(x)                 # optional upsample
        x = torch.cat([x, skip], dim=1)      # concat along channels
        x = self.res1(x, t_emb)
        # Apply attention if present (attention doesn't use t_emb)
        if self.attention is not None:
            x = self.attention(x)
        x = self.res2(x, t_emb)
        return x

class FullDecoderModule(nn.Module):
    '''
    connect all 4 encoder stages(for now)

    '''
    def __init__(self):
        '''
        Passing through Encoder stages 1 by 1
        '''
        super().__init__()
        # Track resolutions: after bottleneck=8, after stage1=8, after stage2=16, after stage3=32, after stage4=64
        self.decoderstage_1 = DecoderStage(in_channels = ds1_in_channels, skip_channels=es4_out_channels, out_channels= ds1_out_channels, upsample=False, resolution=image_size // 8, use_attention=True)
        self.decoderstage_2 = DecoderStage(in_channels = ds2_in_channels, skip_channels=es3_out_channels, out_channels=ds2_out_channels, upsample=True, resolution=image_size // 4, use_attention=True) # Adjusted input channels to include skip connection
        self.decoderstage_3 = DecoderStage(in_channels = ds3_in_channels, skip_channels=es2_out_channels, out_channels=ds3_out_channels, upsample=True, resolution=image_size // 2, use_attention=True) # Adjusted input channels
        self.decoderstage_4 = DecoderStage(in_channels = ds4_in_channels, skip_channels=es1_out_channels, out_channels=ds4_out_channels, upsample=True, resolution=image_size, use_attention=True) # Adjusted input channels
        # Add normalization before final conv to match original
        self.final_norm = nn.GroupNorm(num_groups=n_groupnorm_groups, num_channels=ds4_out_channels)
        self.final_act = nn.SiLU()
        self.finalconv = nn.Conv2d(in_channels = ds4_out_channels, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, bottleneck_output, encoder_outputs, t_emb):#
        # Unpack encoder outputs
        (out_1_enc, skip_1), (out_2_enc, skip_2), (out_3_enc, skip_3), (out_4_enc, skip_4) = encoder_outputs

        # Decoder stages, passing skip connections
        out_1_dec  = self.decoderstage_1(bottleneck_output, skip_4, t_emb) # First decoder stage uses the bottleneck output last encoder output
        #print(f'The shape after Decoder Stage 1 is {out_1_dec.shape}')
        out_2_dec  = self.decoderstage_2(out_1_dec, skip_3, t_emb) # Subsequent stages use previous decoder output and corresponding encoder skip
        #print(f'The shape after Decoder Stage 2 after upsampling is {out_2_dec.shape}')
        out_3_dec  = self.decoderstage_3(out_2_dec, skip_2, t_emb)
        #print(f'The shape after Decoder Stage 3 after upsampling is {out_3_dec.shape}')
        out_4_dec  = self.decoderstage_4(out_3_dec, skip_1, t_emb)
        #print(f'The shape after Encoder Stage 4 after upsampling is {out_4_dec.shape}')
        # Apply normalization and activation before final conv (matching original)
        final_out = self.finalconv(self.final_act(self.final_norm(out_4_dec)))
        #print(f'The shape after final conv is {final_out.shape}')

        return final_out

class FullUNET(nn.Module):
    def __init__(self):
        """
        Full U-Net model with required LR conditioning.
        Concatenates LR image directly with input (assumes same resolution).
        """
        super().__init__()
        
        # Input channels = original input (3) + LR image channels (3) = 6
        input_channels = in_channels + in_channels  # 3 + 3 = 6
        
        self.enc = FullEncoderModule(input_channels=input_channels)
        self.bottleneck = Bottleneck()
        self.dec = FullDecoderModule()
    
    def forward(self, x, t, lq):
        """
        Forward pass with required LR conditioning.
        Args:
            x: (B, C, H, W) Input tensor
            t: (B,) Timestep tensor
            lq: (B, C_lq, H_lq, W_lq) LR image for conditioning (required, same resolution as x)
        Returns:
            out: (B, out_channels, H, W) Output tensor
        """
        # Compute time embedding once for efficiency
        t_emb = sinusoidal_embedding(t)  # (B, timestep_embed_dim)
        
        # Concatenate LR image directly with input along channel dimension
        # Assumes lq has same spatial dimensions as x
        x = torch.cat([x, lq], dim=1)
        
        encoder_outputs = self.enc(x, t_emb)  # with pre-computed time embedding
        (out_1_enc, skip_1), (out_2_enc, skip_2), (out_3_enc, skip_3), (out_4_enc, skip_4) = encoder_outputs
        bottle_neck_output = self.bottleneck(out_4_enc, t_emb)
        out = self.dec(bottle_neck_output, encoder_outputs, t_emb)
        return out

