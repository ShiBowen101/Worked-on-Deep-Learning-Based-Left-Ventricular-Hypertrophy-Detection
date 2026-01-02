import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from Swin_Patch import SwinPatchTST as PatchTST # <-- 核心修改：导入新模型并使用别名
# 从各自的文件中导入
from CNN_transformer import ConvTransformer as YourModel # <-- 核心修改：导入新模型并使用别名
from train_data_pre import get_dataloaders # <-- 导入新的智能加载函数
from sklearn.metrics import accuracy_score, f1_score # <-- 增加f1_score
from visualize import MetricsVisualizer # <-- 导入新的可视化工具

# --- 1. 定义超参数 ---
# 数据路径参数
RAW_DATA_PATH = "data"  # <-- 原始数据文件夹路径
PREPROCESSED_DATA_PATH = "data/preprocessed_ecg_data_strong.npz" # <-- 缓存文件路径

# 数据规格参数
SEQ_LENGTH = 5000
NUM_FEATURES = 12

# ConvTransformer 模型参数
D_MODEL = 64                   # Transformer的嵌入维度
NHEAD = 8                       # 多头注意力头数
NUM_TRANSFORMER_LAYERS = 4      # Transformer编码器层数
DIM_FEEDFORWARD = 128          # Transformer前馈网络维度
DROPOUT = 0.3                   # 统一的Dropout率

# 训练参数
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. 训练和评估循环 (保持不变) ---
def train_loop(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training"):
        X_batch, y_batch = [b.to(device) for b in batch]
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.round(torch.sigmoid(logits)).squeeze()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')  # <-- 计算F1分数
    return avg_loss, accuracy, f1


def val_loop(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            X_batch, y_batch = [b.to(device) for b in batch]
            logits = model(X_batch)
            loss = loss_fn(logits.squeeze(), y_batch)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).squeeze()  # <-- 获取概率用于ROC
            preds = torch.round(probs)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # <-- 收集概率

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')  # <-- 计算F1分数
    return avg_loss, accuracy, f1, all_labels, all_probs  # <-- 返回概率和真实标签


# --- 3. 主函数 ---
def main():
    print(f"Using device: {DEVICE}")

    # 智能加载数据：只需调用一个函数
    train_loader, val_loader = get_dataloaders(
        preprocessed_file_path=PREPROCESSED_DATA_PATH,
        raw_data_path=RAW_DATA_PATH,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        num_features=NUM_FEATURES
    )
    visualizer = MetricsVisualizer(save_dir='training_plots')  # <-- 初始化可视化工具
    # 构建模型
    model = YourModel(
        input_features=NUM_FEATURES,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=1
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    # 损失函数和优化器
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 早停逻辑所需的变量
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # 训练循环
        train_loss, train_acc, train_f1 = train_loop(model, train_loader, optimizer, loss_fn, DEVICE)

        # 验证循环
        val_loss, val_acc, val_f1, y_true_val, y_probs_val = val_loop(model, val_loader, loss_fn, DEVICE)

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # 更新可视化工具的历史记录
        epoch_metrics = {
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc': train_acc, 'val_acc': val_acc,
            'train_f1': train_f1, 'val_f1': val_f1
        }
        visualizer.update(epoch_metrics)

        # 在每个epoch都更新ROC数据 (只在最后一个epoch画图，但数据需要累积)
        # 为了避免内存问题，只在最后一个epoch收集ROC数据
        if epoch == EPOCHS - 1:
            visualizer.update_roc_data(y_true_val, y_probs_val)

        # ... (学习率调度和早停逻辑保持不变) ...

    print("\n训练完成。")
    # 训练结束后，调用plot_all来生成所有图表
    visualizer.plot_all()


if __name__ == "__main__":
    main()
