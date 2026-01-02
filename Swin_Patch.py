# model.py (SwinPatchTST Channel-Independent Version)
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange  # A powerful library for tensor manipulation, highly recommended!


# 如果没有安装，请 pip install einops

# Helper function to create overlapping windows
def window_partition(x, window_size):
    """
    Args:
        x: (B, N, C)  N is the number of patches
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.permute(0, 2, 1, 3).contiguous().view(-1, N // window_size, C)
    return windows


# Helper function to reverse window partition
def window_reverse(windows, window_size, N, B):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        N (int): Number of patches
        B (int): Batch size
    Returns:
        x: (B, N, C)
    """
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ A basic Swin Transformer layer for 1D data. """

    def __init__(self, dim, num_patches, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()  # Placeholder, can be replaced with DropPath
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            attn_mask = self.create_mask()
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def create_mask(self):
        # calculate attention mask for SW-MSA
        N = self.num_patches
        img_mask = torch.zeros((1, N, 1))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for w in w_slices:
            img_mask[:, w, :] = cnt
            cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        N = self.num_patches
        B, L, C = x.shape
        assert L == N, "Input feature has wrong number of patches"

        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, N, B)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# --- The Main Model ---
class SwinPatchTST(nn.Module):
    def __init__(self,
                 seq_len: int, patch_len: int, input_features: int, d_model: int,
                 nhead: int, num_encoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1, num_classes: int = 1,
                 window_size: int = 8):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len
        self.num_patches = (seq_len - self.patch_len) // self.stride + 1

        # Check if num_patches is divisible by window_size
        assert self.num_patches % window_size == 0, \
            f"Number of patches ({self.num_patches}) must be divisible by window_size ({window_size})."

        # --- Shared Components for all channels ---
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # --- Swin Transformer Encoder ---
        self.encoder_layers = nn.ModuleList()
        for i in range(num_encoder_layers):
            self.encoder_layers.append(
                SwinTransformerBlock(
                    dim=d_model,
                    num_patches=self.num_patches,
                    num_heads=nhead,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=dim_feedforward // d_model
                )
            )

        # --- Classification Head ---
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(d_model * input_features),
            nn.Linear(d_model * input_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_features)

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std
        x_norm = x_norm.permute(0, 2, 1)  # (B, C, L)

        channel_outputs = []
        for i in range(x_norm.shape[1]):  # Iterate through channels
            channel_data = x_norm[:, i, :]  # (B, L)
            patches = channel_data.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, N, P)
            embedding = self.patch_embedding(patches)  # (B, N, d_model)
            pos_encoded_input = embedding + self.positional_encoding

            # Pass through Swin Encoder
            encoded_output = pos_encoded_input
            for layer in self.encoder_layers:
                encoded_output = layer(encoded_output)

            # Use the first patch output as channel feature
            channel_feature = encoded_output[:, 0, :]  # (B, d_model)
            channel_outputs.append(channel_feature)

        all_channel_features = torch.stack(channel_outputs, dim=1)  # (B, C, d_model)
        logits = self.head(all_channel_features)
        return logits