

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# --- 1. 设置路径和参数 ---
BASE_DATA_DIR = "data"  # 包含 '0' 和 '1' 文件夹的根目录
OUTPUT_DIR = "processed_data_split"  # 新的输出目录，避免与旧数据混淆

# 数据参数
N_LEADS = 12
N_POINTS = 5000

# 数据增强参数 (只对训练集使用)
AUGMENTATION_FACTOR = 4

# --- 2. 定义预处理和数据增强函数 (与之前相同) ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sampling_rate = 500
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def preprocess_ecg(ecg_data):
    processed_data = butter_bandpass_filter(ecg_data, lowcut=0.5, highcut=45, fs=500)
    processed_data = minmax_scale(processed_data, feature_range=(0, 1), axis=0)
    return processed_data

def augment_data(ecg_data):
    augmented_data = ecg_data.copy()
    choice = random.randint(1, 3)
    if choice == 1:
        noise = np.random.normal(0, random.uniform(0, 0.05), augmented_data.shape)
        augmented_data += noise
    elif choice == 2:
        augmented_data *= random.uniform(0.9, 1.1)
    elif choice == 3:
        shift_amount = random.randint(-int(N_POINTS * 0.05), int(N_POINTS * 0.05))
        if shift_amount != 0:
            augmented_data = np.roll(augmented_data, shift_amount, axis=0)
    return np.clip(augmented_data, 0, 1)

# --- 3. 新的主处理逻辑 ---
def process_files(file_list, augment=False):
    """
    一个通用的函数，用于处理一个文件列表。
    :param file_list: 一个包含 (filepath, label) 元组的列表。
    :param augment: 是否对这批文件进行数据增强。
    :return: 处理好的数据和标签列表。
    """
    all_data = []
    all_labels = []

    for filepath, label in tqdm(file_list, desc=f"Processing files (augment={augment})"):
        try:
            df = pd.read_excel(filepath, header=None)
            ecg_raw_data = df.iloc[4:5004, :N_LEADS].values.astype(np.float32)

            if ecg_raw_data.shape != (N_POINTS, N_LEADS):
                print(f"\n警告：文件 {os.path.basename(filepath)} 形状不正确，已跳过。")
                continue

            # 预处理是所有数据都必须做的
            ecg_processed = preprocess_ecg(ecg_raw_data)

            # 添加原始（已预处理）数据
            all_data.append(ecg_processed)
            all_labels.append(label)

            # 只对指定的数据集进行增强
            if augment:
                for _ in range(AUGMENTATION_FACTOR):
                    ecg_augmented = augment_data(ecg_processed)
                    all_data.append(ecg_augmented)
                    all_labels.append(label)
        except Exception as e:
            print(f"\n处理文件 {os.path.basename(filepath)} 时发生错误: {e}")

    return np.array(all_data, dtype=np.float32), np.array(all_labels, dtype=np.int32)

def main():
    print("开始数据准备和划分...")

    # --- 步骤1: 收集所有原始文件路径和标签 ---
    all_files = []
    for label in ["0", "1"]:
        label_dir = os.path.join(BASE_DATA_DIR, label)
        if not os.path.isdir(label_dir):
            print(f"警告：找不到文件夹 {label_dir}")
            continue
        for filename in os.listdir(label_dir):
            if filename.endswith('.xlsx'):
                filepath = os.path.join(label_dir, filename)
                all_files.append((filepath, int(label)))

    print(f"总共找到 {len(all_files)} 个原始文件。")
    if not all_files:
        print("错误：未找到任何数据文件，请检查BASE_DATA_DIR。")
        return

    # --- 步骤2: 在文件层面进行分层划分 ---
    # 提取文件路径和标签用于划分
    filepaths = [item[0] for item in all_files]
    labels = [item[1] for item in all_files]

    # 第一次划分：分出80%作为训练集，20%作为临时集（包含验证和测试）
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        filepaths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 第二次划分：将临时集对半分为验证集和测试集
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    train_file_list = list(zip(train_files, train_labels))
    val_file_list = list(zip(val_files, val_labels))
    test_file_list = list(zip(test_files, test_labels))

    print("\n文件划分完成:")
    print(f"  - 训练集文件数: {len(train_file_list)}")
    print(f"  - 验证集文件数: {len(val_file_list)}")
    print(f"  - 测试集文件数: {len(test_file_list)}")

    # --- 步骤3: 分别处理三个数据集 ---
    print("\n开始处理训练集 (带数据增强)...")
    X_train, y_train = process_files(train_file_list, augment=True)

    print("\n开始处理验证集 (无数据增强)...")
    X_val, y_val = process_files(val_file_list, augment=False)

    print("\n开始处理测试集 (无数据增强)...")
    X_test, y_test = process_files(test_file_list, augment=False)

    # --- 步骤4: 打乱并保存 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    def shuffle_and_save(X, y, name, output_dir):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        print(f"\n{name} - 最终形状: X={X.shape}, y={y.shape}")

        np.save(os.path.join(output_dir, f"{name}_X.npy"), X)
        np.save(os.path.join(output_dir, f"{name}_y.npy"), y)
        print(f"已保存 {name} 数据到 {output_dir}")

    shuffle_and_save(X_train, y_train, "train", OUTPUT_DIR)
    shuffle_and_save(X_val, y_val, "val", OUTPUT_DIR)
    shuffle_and_save(X_test, y_test, "test", OUTPUT_DIR)

    print("\n所有数据处理和保存完毕！")


if __name__ == "__main__":
    main()