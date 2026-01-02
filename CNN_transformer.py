# model.py (ConvTransformer Hybrid Version)
import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class CNNFeatureExtractor(nn.Module):
    """
    一个高效的CNN模块，使用深度可分离卷积提取特征并降采样。
    """

    def __init__(self, input_features: int, output_dim: int, dropout: float = 0.2):
        super().__init__()

        # 定义一系列的卷积块
        self.blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(input_features, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 5000 -> 2500
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2500 -> 1250
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 1250 -> 312 (approx)
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: (batch_size, seq_len, input_features)
        # Conv1d 需要 (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)

        # 通过CNN块
        x = self.blocks(x)

        # 转换回 (batch_size, new_seq_len, output_dim) 以适配Transformer
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    """标准的正弦/余弦位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConvTransformer(nn.Module):
    """
    CNN-Transformer 混合模型
    """

    def __init__(self,
                 input_features: int,
                 d_model: int,
                 nhead: int,
                 num_transformer_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 num_classes: int = 1):
        super().__init__()

        # 1. CNN 特征提取器
        self.cnn_feature_extractor = CNNFeatureExtractor(
            input_features=input_features,
            output_dim=d_model,  # 让CNN的输出维度与Transformer的d_model匹配
            dropout=dropout
        )

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer 编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers
        )

        # 4. 分类头
        self.d_model = d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # 我们使用一个可学习的 [CLS] token 来聚合序列信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_features)

        # 1. 通过CNN提取特征并降采样
        # 输出 cnn_features shape: (batch_size, new_seq_len, d_model)
        cnn_features = self.cnn_feature_extractor(x)

        # 2. 添加 [CLS] token
        # 获取batch_size
        B = cnn_features.shape[0]
        # 扩展cls_token以匹配batch_size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 将cls_token拼接到序列的开头
        x_with_cls = torch.cat((cls_tokens, cnn_features), dim=1)  # (B, 1 + new_seq_len, d_model)

        # 3. 添加位置编码
        x_with_pos = self.pos_encoder(x_with_cls)

        # 4. 通过 Transformer
        transformer_output = self.transformer_encoder(x_with_pos)  # (B, 1 + new_seq_len, d_model)

        # 5. 提取 [CLS] token 的输出作为整个序列的表征
        cls_output = transformer_output[:, 0, :]  # (B, d_model)

        # 6. 通过分类头得到最终结果
        logits = self.classifier(cls_output)

        return logits