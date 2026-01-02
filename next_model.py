# model_hybrid.py

import torch
import torch.nn as nn


class ECG_CNN_LSTM(nn.Module):
    """
    一个CNN-LSTM混合模型，用于ECG分类。
    - CNN部分负责从原始信号中提取高级形态学特征。
    - LSTM部分负责分析这些特征序列中的时间依赖关系。
    """

    def __init__(self, n_leads=12, n_classes=1, cnn_filters=64, lstm_hidden_size=128):
        super(ECG_CNN_LSTM, self).__init__()

        # --- 1. CNN特征提取主干 ---
        # 我们使用一个简化的、高效的CNN结构作为特征提取器
        self.cnn_backbone = nn.Sequential(
            # Block 1
            nn.Conv1d(n_leads, cnn_filters, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            # Block 2
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            # Block 3
            nn.Conv1d(cnn_filters * 2, cnn_filters * 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(cnn_filters * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        # --- 2. LSTM序列分析器 ---
        # LSTM的输入特征维度等于CNN最后一个卷积层的输出通道数
        lstm_input_size = cnn_filters * 4

        # 使用双向LSTM来捕捉前后文信息
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,  # 堆叠2层LSTM以增加模型深度
            batch_first=True,  # 输入输出格式为 (batch, seq_len, features)
            bidirectional=True,  # 使用双向LSTM
            dropout=0.3  # 在LSTM层之间加入Dropout
        )

        # --- 3. 分类器 ---
        # 分类器的输入维度是LSTM隐藏层大小的两倍（因为是双向的）
        classifier_input_size = lstm_hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x的输入形状: (batch, length, channels)

        # 1. 通过CNN主干
        # permute to: (batch, channels, length) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn_backbone(x)  # 输出形状: (batch, cnn_output_channels, new_length)

        # 2. 调整形状以适应LSTM
        # permute to: (batch, new_length, cnn_output_channels) for LSTM
        x = x.permute(0, 2, 1)

        # 3. 通过LSTM
        # lstm_out 形状: (batch, new_length, hidden_size * 2)
        # self.lstm的输出是一个元组 (all_outputs, (last_hidden_state, last_cell_state))
        lstm_out, _ = self.lstm(x)

        # 4. 提取LSTM的最后一个时间步的输出作为整个序列的表示
        # last_time_step 形状: (batch, hidden_size * 2)
        last_time_step = lstm_out[:, -1, :]

        # 5. 通过分类器得到最终结果
        logits = self.classifier(last_time_step)

        return logits