# train_ultimate.py
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import os
import argparse

from RCST import TsaiResNetSwinTransformer as YourModel
from RCST_data import get_dataloaders

# from visualize import MetricsVisualizer # 如果您有可视化工具，取消此行注释

# --- 1. 定义超参数 ---
RAW_DATA_PATH = "data"
PREPROCESSED_DATA_PATH = "data/preprocessed_ecg_data_RCST.npz"
CHECKPOINT_DIR = "checkpoints_ultimate"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

SEQ_LENGTH = 5000
NUM_FEATURES = 12

# 模型参数
D_MODEL = 128
NHEAD = 8
NUM_SWIN_LAYERS = 4
WINDOW_SIZE = 8  # ResNet降采样后序列长度为312，8是其因子
DIM_FEEDFORWARD = 256
DROPOUT = 0.2

# 训练参数
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 20
WARMUP_EPOCHS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def main(args):
    print(f"Using device: {DEVICE}")
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    train_loader, val_loader = get_dataloaders(
        preprocessed_file_path=PREPROCESSED_DATA_PATH, raw_data_path=RAW_DATA_PATH,
        batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH, num_features=NUM_FEATURES
    )
    # visualizer = MetricsVisualizer(save_dir='training_plots_ultimate')

    model = YourModel(
        input_channels=NUM_FEATURES, seq_len=SEQ_LENGTH, d_model=D_MODEL,
        nhead=NHEAD, num_swin_layers=NUM_SWIN_LAYERS, window_size=WINDOW_SIZE,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, num_classes=1
    ).to(DEVICE)

    for param in model.parameters(): param.requires_grad = False
    for param in model.projection.parameters(): param.requires_grad = True
    for param in model.swin_encoder.parameters(): param.requires_grad = True
    for param in model.head_norm.parameters(): param.requires_grad = True
    for param in model.head.parameters(): param.requires_grad = True
    print("解冻 tsai ResNet backbone 的最后两个 block...")
    for param in model.feature_extractor[-1].parameters(): param.requires_grad = True
    for param in model.feature_extractor[-2].parameters(): param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    start_epoch, best_val_f1 = 0, 0.0
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_val_f1']
        print(f"已加载检查点。将从 Epoch {start_epoch + 1} 开始。")

    epochs_no_improve = 0
    print("\n开始训练...")
    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc, train_f1 = train_loop(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_acc, val_f1, y_true_val, y_probs_val = val_loop(model, val_loader, loss_fn, DEVICE)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # visualizer.update(...)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(), 'best_val_f1': best_val_f1}, CHECKPOINT_PATH)
            print(f"Validation F1 improved. Saved checkpoint to '{CHECKPOINT_PATH}'")
            # visualizer.update_roc_data(...)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Validation F1 did not improve for {EARLY_STOPPING_PATIENCE} epochs. Early stopping.")
            break

        scheduler.step()

    print(f"\n训练完成。\nBest validation F1 recorded: {best_val_f1:.4f}")
    # visualizer.plot_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TsaiResNet-SwinTransformer Training")
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    main(args)