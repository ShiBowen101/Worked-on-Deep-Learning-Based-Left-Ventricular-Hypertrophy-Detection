# data_utils_1d.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tsaug import TimeWarp, AddNoise, Drift


def scale_augmentation(x, max_scale=1.2):
    scaling_factor = np.random.uniform(1 / max_scale, max_scale)
    return x * scaling_factor


class ECG1DDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool):
        self.X = X
        self.y = y
        self.is_train = is_train

        if self.is_train:
            self.tsaug_augmenter = (
                    AddNoise(scale=0.015) @ 0.5
                    + TimeWarp(n_speed_change=5, max_speed_ratio=1.5) @ 0.5
                    + Drift(max_drift=0.05, n_drift_points=3) @ 0.3
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_sample = self.X[idx].copy()
        y_sample = self.y[idx]

        if self.is_train:
            x_sample = self.tsaug_augmenter.augment(x_sample)
            if np.random.rand() < 0.5:
                x_sample = scale_augmentation(x_sample)

        x_tensor = torch.from_numpy(x_sample).float().permute(1, 0)
        y_tensor = torch.tensor(y_sample, dtype=torch.float32)

        return x_tensor, y_tensor


def _process_raw_data_and_save(raw_data_path, output_file_path, seq_len, num_features):
    print("=" * 50)
    print(f"开始从原始数据文件夹 '{raw_data_path}' 进行预处理...")
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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n预处理完成。正在将结果保存到 '{output_file_path}'...")
    np.savez_compressed(output_file_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("数据保存成功！")

    return X_train, y_train, X_val, y_val


def get_dataloaders(preprocessed_file_path, raw_data_path, batch_size, seq_len, num_features):
    if os.path.exists(preprocessed_file_path):
        print(f"发现预处理文件 '{preprocessed_file_path}'，正在直接加载...")
        data = np.load(preprocessed_file_path)
        X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']
    else:
        X_train, y_train, X_val, y_val = _process_raw_data_and_save(raw_data_path, preprocessed_file_path, seq_len,
                                                                    num_features)

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")

    train_dataset = ECG1DDataset(X_train, y_train, is_train=True)
    val_dataset = ECG1DDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("DataLoader创建完成，ECG将作为1D时间序列处理。")
    return train_loader, val_loader