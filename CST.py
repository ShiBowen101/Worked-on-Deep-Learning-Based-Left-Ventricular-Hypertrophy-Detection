# model.py (CNN - Swin Transformer Hybrid Version)
import torch
from torch import nn


# --- Helper Functions and Classes for Swin Transformer ---
def window_partition(x, window_size):
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.permute(0, 2, 1, 3).contiguous().view(-1, N // window_size,
                                                      C)  # Mistake here, should be (B * num_windows, window_size, C)
    # Correct implementation:
    windows = x.view(B, N // window_size, window_size, C).permute(0, 2, 1, 3).contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, N, B):
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        # ... (代码与SwinPatchTST中的版本完全相同) ...
        super().__init__()
        self.dim = dim
        self.window_size = window_size[0] if isinstance(window_size, tuple) else window_size
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
    def __init__(self, dim, num_patches, num_heads, window_size=8, shift_size=0, mlp_ratio=4., **kwargs):

        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size,), num_heads=num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, dim), nn.Dropout(0.1)
        )

        if self.shift_size > 0:
            H = self.num_patches
            img_mask = torch.zeros((1, H, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                img_mask[:, h, :] = cnt
                cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H = self.num_patches
        B, L, C = x.shape
        assert L == H, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, B)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- The Main CNN-SwinTransformer Model ---
class CnnSwinTransformer(nn.Module):
    def __init__(self,
                 input_features: int, seq_len: int,
                 cnn_output_dim: int,
                 d_model: int, nhead: int, num_swin_layers: int,
                 window_size: int,
                 dim_feedforward: int, dropout: float = 0.1, num_classes: int = 1):
        super().__init__()

        # 1. CNN Feature Extractor with light pooling
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(input_features, 128,kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, cnn_output_dim, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)  # Light pooling: 5000 -> 1250
        )

        # Calculate the sequence length after CNN
        cnn_output_seq_len = seq_len // 4

        # 2. Projection layer (if cnn_output_dim is not d_model)
        self.projection = nn.Linear(cnn_output_dim, d_model) if cnn_output_dim != d_model else nn.Identity()

        # 3. Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, cnn_output_seq_len + 1, d_model))

        # 4. Swin Transformer Encoder
        self.swin_encoder = nn.ModuleList([
            SwinTransformerBlock(
                dim=d_model,
                num_patches=cnn_output_seq_len,
                num_heads=nhead,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=dim_feedforward // d_model
            ) for i in range(num_swin_layers)
        ])

        # 5. Classification Head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C_in)

        # 1. CNN Feature Extraction
        x = x.permute(0, 2, 1)  # (B, C_in, L)
        x = self.cnn_extractor(x)  # (B, cnn_output_dim, L_new)
        x = x.permute(0, 2, 1)  # (B, L_new, cnn_output_dim)

        # 2. Projection to d_model
        x = self.projection(x)  # (B, L_new, d_model)

        # 3. Prepend CLS token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + L_new, d_model)

        # 4. Add positional encoding
        x = x + self.pos_encoder

        # 5. Pass through Swin Transformer
        # Note: We pass only sequence tokens to Swin, then add back CLS
        seq_tokens = x[:, 1:, :]
        for layer in self.swin_encoder:
            seq_tokens = layer(seq_tokens)

        # 6. Extract CLS token output for classification
        cls_output = x[:, 0, :]  # Use the original CLS token embedding

        # A more common approach is to let CLS token also pass through a standard MHA layer
        # For simplicity, we just use the CLS token and process the sequence.
        # Let's refine this: the CLS token should also interact.
        # A simpler way for now: average pooling on swin output

        # --- Refined Forward Pass ---
        # 5. (Alternative & Simpler) Pass through Swin Transformer
        # We need to adapt Swin to handle the CLS token.
        # Or, a simpler and robust way: Don't use CLS token with Swin. Use pooling.

        # --- Let's rewrite forward pass for simplicity and robustness ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C_in)
        x = x.permute(0, 2, 1)  # (B, C_in, L)
        x = self.cnn_extractor(x)  # (B, cnn_output_dim, L_new)
        x = x.permute(0, 2, 1)  # (B, L_new, cnn_output_dim)

        x = self.projection(x)  # (B, L_new, d_model)

        # We'll use a simpler positional encoding here for variable length after CNN
        x = x + nn.Parameter(torch.randn(1, x.size(1), x.size(2)).to(x.device))

        for layer in self.swin_encoder:
            x = layer(x)

        # Use mean pooling over the sequence dimension
        x = x.mean(dim=1)  # (B, d_model)

        x = self.head_norm(x)
        logits = self.head(x)

        return logits