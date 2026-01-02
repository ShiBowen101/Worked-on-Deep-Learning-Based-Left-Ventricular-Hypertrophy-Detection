import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from Swin_Patch import SwinPatchTST as PatchTST # <-- 核心修改：导入新模型并使用别名
# 从各自的文件中导入
from PatchTSTModel import PatchTST_ChannelIndependent
from train_data_pre import get_dataloaders # <-- 导入新的智能加载函数

# --- 1. 定义超参数 ---
# 数据路径参数
RAW_DATA_PATH = "data"  # <-- 原始数据文件夹路径
PREPROCESSED_DATA_PATH = "data/preprocessed_ecg_data_strong.npz" # <-- 缓存文件路径

# 数据规格参数
SEQ_LENGTH = 5000
NUM_FEATURES = 12

# 模型参数
# --- 注意: num_patches = (SEQ_LENGTH - PATCH_LEN) / PATCH_LEN + 1 必须能被 WINDOW_SIZE 整除 ---
PATCH_LEN = 16          # num_patches = 312
WINDOW_SIZE = 8         # 312 % 8 == 0, 所以这个组合是有效的
D_MODEL = 32
NHEAD = 8               # Swin-Transformer中, nhead 通常和 dim 相关, d_model=32, nhead=8 可能过大, 建议 nhead=4
NUM_ENCODER_LAYERS = 4  # Swin Block通常会堆叠更多层
DIM_FEEDFORWARD = 64
DROPOUT = 0.3

# 训练参数
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. 训练和评估循环 (保持不变) ---
def train_loop(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

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
    return avg_loss, accuracy


def val_loop(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            X_batch, y_batch = [b.to(device) for b in batch]

            logits = model(X_batch)
            loss = loss_fn(logits.squeeze(), y_batch)

            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(logits)).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


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

    # 构建模型
    model = PatchTST(
        seq_len=SEQ_LENGTH,
        patch_len=PATCH_LEN,
        input_features=NUM_FEATURES,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=1,
        window_size=WINDOW_SIZE  # <-- 传入新的超参数
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
        # 训练和验证循环
        train_loss, train_acc = train_loop(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_acc = val_loop(model, val_loader, loss_fn, DEVICE)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_patchtst_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"验证性能连续 {EARLY_STOPPING_PATIENCE} 个epoch没有改善，触发早停。")
            break

    print("\n训练完成。")
    print(f"最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
