# data_utils.py (手动实现幅度缩放 + tsaug 增强版)
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# 导入tsaug库中真实存在的增强方法
from tsaug import TimeWarp, AddNoise, Drift


class ECGDataset(Dataset):
    """
    自定义ECG数据集，并在训练时应用数据增强。
    包含手动实现的幅度缩放和tsaug库的其他增强。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool):
        """
        Args:
            X (np.ndarray): 数据样本.
            y (np.ndarray): 标签.
            is_train (bool): 标志位，决定是否应用数据增强。
        """
        self.X = X
        self.y = y
        self.is_train = is_train

        # 定义一个不包含Scale的tsaug增强器
        # 我们将手动实现幅度缩放
        self.tsaug_augmenter = (
                AddNoise(scale=0.015) @ 0.5  # 50%的概率增加高斯噪声
                + TimeWarp(n_speed_change=5, max_speed_ratio=1.5) @ 0.5  # 50%的概率进行时间扭曲
                + Drift(max_drift=0.05, n_drift_points=3) @ 0.3  # 30%的概率进行轻微的基线漂移
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_sample = self.X[idx].copy()  # 使用 .copy() 避免修改原始数据
        y_sample = self.y[idx]

        # 只对训练集应用数据增强
        if self.is_train:
            # ==================== 1. 手动实现幅度缩放 ====================
            # 以50%的概率应用幅度缩放
            if np.random.rand() < 0.5:
                # 生成一个在[0.9, 1.1]范围内的随机缩放因子
                scaling_factor = np.random.uniform(0.9, 1.1)
                x_sample = x_sample * scaling_factor

            # ==================== 2. 应用 tsaug 的其他增强 ====================
            # tsaug需要 (N, L, C) 格式的输入
            x_sample = self.tsaug_augmenter.augment(x_sample[np.newaxis, :, :])[0]

            # 转换为torch.Tensor
            return torch.tensor(x_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.float32)

        # 对于验证集，直接返回原始数据
        return torch.tensor(x_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.float32)


# _process_raw_data_and_save 和 get_dataloaders 函数保持不变
# 您可以从之前的回答中复制它们，它们的功能是正确的。
# 为保证完整性，我再次贴出 get_dataloaders 的代码。

def _process_raw_data_and_save(raw_data_path: str, output_file_path: str, seq_len: int, num_features: int) -> tuple:
    # ... (这个函数保持不变) ...
    print("=" * 50)
    print("开始从原始Excel文件进行数据预处理...")
    print(f"数据源: {raw_data_path}")
    print("=" * 50)

    all_data, all_labels = [], []
    class_folders = sorted([d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))])
    class_to_label = {name: i for i, name in enumerate(class_folders)}

    for folder_name, label in class_to_label.items():
        folder_path = os.path.join(raw_data_path, folder_name)
        file_list = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.csv'))]
        for filename in tqdm(file_list, desc=f"加载 {folder_name}"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_excel(file_path, header=None, skiprows=4)
                if df.values.shape == (seq_len, num_features):
                    all_data.append(df.values)
                    all_labels.append(label)
            except Exception as e:
                print(f"警告: 读取文件 {filename} 失败: {e}")

    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    print(f"\n预处理完成。正在将结果保存到 '{output_file_path}'...")
    np.savez_compressed(output_file_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("数据保存成功！")

    return X_train, y_train, X_val, y_val


def get_dataloaders(preprocessed_file_path: str,
                    raw_data_path: str,
                    batch_size: int,
                    seq_len: int,
                    num_features: int) -> tuple[DataLoader, DataLoader]:
    if os.path.exists(preprocessed_file_path):
        print(f"发现预处理文件 '{preprocessed_file_path}'，正在直接加载...")
        data = np.load(preprocessed_file_path)
        X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']
    else:
        print(f"未发现预处理文件。将从原始数据文件夹 '{raw_data_path}' 开始处理。")
        X_train, y_train, X_val, y_val = _process_raw_data_and_save(
            raw_data_path=raw_data_path,
            output_file_path=preprocessed_file_path,
            seq_len=seq_len,
            num_features=num_features
        )

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")

    train_dataset = ECGDataset(X_train, y_train, is_train=True)
    val_dataset = ECGDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print("DataLoader创建完成，已为训练集启用数据增强。")
    return train_loader, val_loader