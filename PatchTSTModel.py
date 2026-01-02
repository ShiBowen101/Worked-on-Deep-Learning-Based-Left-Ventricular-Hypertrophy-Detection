import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

# model.py (Channel-Independent Version)
import torch
from torch import nn


class PatchTST_ChannelIndependent(nn.Module):
    def __init__(self,
                 seq_len: int, patch_len: int, input_features: int, d_model: int,
                 nhead: int, num_encoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1, num_classes: int = 1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len
        self.num_patches = (seq_len - self.patch_len) // self.stride + 1

        # --- Shared Components for all channels ---
        # 1. Patching and Embedding Layer (shared)
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # 2. Positional Encoding (shared)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # 3. Transformer Encoder (shared)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        # --- Classification Head ---
        # 4. Head that mixes features from all channels
        # Takes the flattened output from all channels
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

        # Instance Normalization (per-sample)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        # Permute to (batch_size, input_features, seq_len) for channel-wise processing
        x_norm = x_norm.permute(0, 2, 1)

        batch_size, num_features, seq_len = x_norm.shape

        # Process each channel independently
        channel_outputs = []
        for i in range(num_features):
            channel_data = x_norm[:, i, :]  # (batch_size, seq_len)

            # Patching: (batch_size, num_patches, patch_len)
            patches = channel_data.unfold(dimension=-1, size=self.patch_len, step=self.stride)

            # Embedding: (batch_size, num_patches, d_model)
            embedding = self.patch_embedding(patches)

            # Add positional encoding
            pos_encoded_input = embedding + self.positional_encoding

            # Transformer Encoder
            transformer_output = self.transformer_encoder(pos_encoded_input)  # (batch, num_patches, d_model)

            # We take the output of the first patch (like a [CLS] token) as the feature for this channel
            channel_feature = transformer_output[:, 0, :]  # (batch, d_model)
            channel_outputs.append(channel_feature)

        # Concatenate features from all channels
        # Shape: (batch_size, num_features, d_model)
        all_channel_features = torch.stack(channel_outputs, dim=1)

        # Flatten and pass to the head
        logits = self.head(all_channel_features)

        return logits
