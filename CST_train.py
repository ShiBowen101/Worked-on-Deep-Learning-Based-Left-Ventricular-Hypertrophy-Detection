
import torch
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import os
import argparse

# 从各自的文件中导入
from CST import CnnSwinTransformer as YourModel
from train_data_pre import get_dataloaders
from visualize import MetricsVisualizer

# --- 1. 定义超参数 ---
# 数据路径参数
RAW_DATA_PATH = "data"
PREPROCESSED_DATA_PATH = "data_large/preprocessed_ecg_data_strong.npz"
CHECKPOINT_DIR = "checkpoints_CST_final"  # 建议为每次实验设置不同的检查点目录
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# 数据规格
SEQ_LENGTH = 5000
NUM_FEATURES = 12

# 模型参数
CNN_OUTPUT_DIM = 192
D_MODEL = 192
NHEAD = 8
NUM_SWIN_LAYERS = 6
WINDOW_SIZE = 10
DIM_FEEDFORWARD = 384
DROPOUT = 0.2

# 训练参数
EPOCHS = 120
BATCH_SIZE = 16
BASE_LEARNING_RATE = 5e-5  # 预热结束后的基础学习率
WARMUP_EPOCHS = 5  # 预热的轮数
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 120

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. 训练和评估循环 (完全保持不变) ---
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
    f1 = f1_score(all_labels, all_preds, average='binary')
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
            probs = torch.sigmoid(logits).squeeze()
            preds = torch.round(probs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    return avg_loss, accuracy, f1, all_labels, all_probs


# --- 3. 主函数 ---
def main(args):
    print(f"Using device: {DEVICE}")

    # 在训练开始前就创建好检查点目录
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    train_loader, val_loader = get_dataloaders(
        preprocessed_file_path=PREPROCESSED_DATA_PATH,
        raw_data_path=RAW_DATA_PATH,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        num_features=NUM_FEATURES
    )
    visualizer = MetricsVisualizer(save_dir='training_plots')

    model = YourModel(
        input_features=NUM_FEATURES, seq_len=SEQ_LENGTH,
        cnn_output_dim=CNN_OUTPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
        num_swin_layers=NUM_SWIN_LAYERS, window_size=WINDOW_SIZE,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, num_classes=1
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 学习率调度器
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    start_epoch = 0

    best_val_f1 = 0.0

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"发现检查点 '{CHECKPOINT_PATH}'，正在加载并继续训练...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_val_f1']  # 恢复最佳F1分数
        if 'metrics_history' in checkpoint:
            visualizer.load_history(checkpoint['metrics_history'])
        print(f"已成功加载检查点。将从 Epoch {start_epoch + 1} 开始训练。")
    else:
        if args.resume:
            print(f"警告: 指定了 --resume，但未找到检查点 '{CHECKPOINT_PATH}'。将从头开始训练。")
        else:
            print("将从头开始训练。")

    epochs_no_improve = 0

    print("\n开始训练...")
    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc, train_f1 = train_loop(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_acc, val_f1, y_true_val, y_probs_val = val_loop(model, val_loader, loss_fn, DEVICE)

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        epoch_metrics = {
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc': train_acc, 'val_acc': val_acc,
            'train_f1': train_f1, 'val_f1': val_f1
        }
        visualizer.update(epoch_metrics)

        # 基于 best_val_f1 进行保存和早停
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,  # 保存最佳F1
                'metrics_history': visualizer.history
            }
            torch.save(checkpoint_data, CHECKPOINT_PATH)
            print(f"Validation F1 improved to {best_val_f1:.4f}. Saved checkpoint to '{CHECKPOINT_PATH}'")
            visualizer.update_roc_data(y_true_val, y_probs_val)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Validation F1 did not improve for {EARLY_STOPPING_PATIENCE} epochs. Early stopping.")
            break

        # scheduler.step() 应该在optimizer.step()之后，通常放在每个epoch的末尾
        scheduler.step()

    print("\n训练完成。")
    print(f"Best validation F1 recorded: {best_val_f1:.4f}")

    if os.path.exists(CHECKPOINT_PATH):
        print("加载性能最佳的模型用于最终可视化...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'metrics_history' in checkpoint:
            visualizer.load_history(checkpoint['metrics_history'])

    visualizer.plot_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Classification Training Script")
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()
    main(args)