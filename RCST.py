# model_ultimate.py
import torch
from torch import nn
from einops import rearrange

# 1. 导入tsai的ResNet模型
from tsai.models.ResNet import ResNet as TsaiResNet


# 2. 包含我们之前已经验证过的1D Swin Transformer Block
def window_partition(x, window_size):
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.view(B, N // window_size, window_size, C).permute(0, 2, 1, 3).contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, N, B):
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
    return x


class WindowAttention(nn.Module):
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
    def __init__(self, dim, num_patches, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp_hidden_dim, dim), nn.Dropout(drop))

        if self.shift_size > 0:
            H = self.num_patches
            img_mask = torch.zeros((1, H, 1))
            h_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices: img_mask[:, h, :] = cnt; cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H = self.num_patches
        B, L, C = x.shape
        assert L == H, f"Input feature has wrong number of patches. Expected {H}, got {L}"
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, H, B)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TsaiResNetSwinTransformer(nn.Module):
    def __init__(self,
                 input_channels: int,
                 seq_len: int,
                 d_model: int,
                 nhead: int,
                 num_swin_layers: int,
                 window_size: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 num_classes: int = 1):
        super().__init__()

        tsai_resnet = TsaiResNet(c_in=input_channels, c_out=num_classes)
        self.feature_extractor = tsai_resnet[0]

        # 获取ResNet主干网络输出的特征维度
        # tsai的ResNet是一个Sequential，我们可以通过访问最后一个block的最后一个卷积层来获取输出通道数
        self.resnet_output_dim = tsai_resnet[0][-1][-1].conv3.out_channels

        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, seq_len)
            dummy_output = self.feature_extractor(dummy_input)
            raw_output_seq_len = dummy_output.shape[2]

        print(f"tsai ResNet backbone output: channels={self.resnet_output_dim}, sequence length={raw_output_seq_len}")

        if raw_output_seq_len % window_size != 0:
            self.resnet_output_seq_len = (raw_output_seq_len // window_size) * window_size
            print(
                f"Warning: Sequence length after ResNet ({raw_output_seq_len}) is not divisible by window_size ({window_size}). It will be truncated to {self.resnet_output_seq_len}.")
        else:
            self.resnet_output_seq_len = raw_output_seq_len
            print("Swin Transformer window constraint is satisfied.")

        self.projection = nn.Linear(self.resnet_output_dim, d_model)

        self.swin_encoder = nn.ModuleList([
            SwinTransformerBlock(
                dim=d_model, num_patches=self.resnet_output_seq_len, num_heads=nhead,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=dim_feedforward // d_model, drop=dropout, attn_drop=dropout
            ) for i in range(num_swin_layers)
        ])

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)

        if features.shape[2] > self.resnet_output_seq_len:
            features = features[:, :, :self.resnet_output_seq_len]

        features = features.permute(0, 2, 1)
        projected_features = self.projection(features)

        encoded_features = projected_features
        for layer in self.swin_encoder:
            encoded_features = layer(encoded_features)

        pooled_output = encoded_features.mean(dim=1)

        norm_output = self.head_norm(pooled_output)
        logits = self.head(norm_output)

        return logits