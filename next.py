# train_hybrid.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 导入新的混合模型
from next_model import ECG_CNN_LSTM


# --- 1. 自定义数据集类 (保持不变) ---
class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- 2. 辅助函数、训练和评估函数 (保持不变) ---
def specificity_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1): return 1.0 if np.unique(y_true)[0] == 0 else 0.0
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="Training")
    for i, (features, labels) in enumerate(pbar):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{correct_predictions / total_samples:.4f}')
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs).squeeze()
            all_preds_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / len(dataloader)
    all_preds, all_labels_np = (np.array(all_preds_probs) > 0.5).astype(int), np.array(all_labels)
    metrics = {
        'loss': avg_loss, 'accuracy': accuracy_score(all_labels_np, all_preds),
        'f1': f1_score(all_labels_np, all_preds, zero_division=0),
        'precision': precision_score(all_labels_np, all_preds, zero_division=0),
        'recall': recall_score(all_labels_np, all_preds, zero_division=0),
        'specificity': specificity_score(all_labels_np, all_preds),
        'auc': roc_auc_score(all_labels_np, all_preds_probs) if len(np.unique(all_labels_np)) > 1 else 0.5
    }
    return metrics, all_labels_np, all_preds


# --- 3. 可视化函数 (保持不变) ---
def plot_metrics(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 10));
    plt.subplot(2, 2, 1);
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss');
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss');
    plt.title('Training and Validation Loss');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend()
    plt.subplot(2, 2, 2);
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy');
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy');
    plt.title('Training and Validation Accuracy');
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy');
    plt.legend()
    plt.subplot(2, 2, 3);
    plt.plot(epochs, history['val_f1'], 'g-', label='Validation F1 Score');
    plt.title('Validation F1 Score');
    plt.xlabel('Epochs');
    plt.ylabel('F1 Score');
    plt.legend()
    plt.subplot(2, 2, 4);
    plt.plot(epochs, history['val_auc'], 'm-', label='Validation AUC');
    plt.title('Validation AUC');
    plt.xlabel('Epochs');
    plt.ylabel('AUC');
    plt.legend()
    plt.tight_layout();
    plt.savefig(save_path);
    plt.close();
    print(f"Metrics plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred);
    plt.figure(figsize=(8, 6));
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names);
    plt.title('Confusion Matrix');
    plt.ylabel('True Label');
    plt.xlabel('Predicted Label');
    plt.savefig(save_path);
    plt.close();
    print(f"Confusion matrix saved to {save_path}")


# --- 4. 主训练逻辑 ---
def main(args):
    # --- 设置 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.save_dir, f"hybrid_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"All outputs will be saved in: {run_dir}\nUsing device: {device}")

    # --- 数据加载 ---
    print("Loading pre-split data...")
    X_train = np.load(os.path.join(args.data_dir, "train_X.npy"));
    y_train = np.load(os.path.join(args.data_dir, "train_y.npy"))
    X_val = np.load(os.path.join(args.data_dir, "val_X.npy"));
    y_val = np.load(os.path.join(args.data_dir, "val_y.npy"))
    X_test = np.load(os.path.join(args.data_dir, "test_X.npy"));
    y_test = np.load(os.path.join(args.data_dir, "test_y.npy"))

    train_dataset, val_dataset, test_dataset = ECGDataset(X_train, y_train), ECGDataset(X_val, y_val), ECGDataset(
        X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Dataset loaded: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # --- **核心改动: 初始化混合模型** ---
    model = ECG_CNN_LSTM(n_leads=X_train.shape[2], n_classes=1).to(device)
    print(f"Model Initialized: {model.__class__.__name__}")

    # --- 训练设置 ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    # --- 训练循环 ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}
    best_val_auc = 0.0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss);
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss']);
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1']);
        history['val_auc'].append(val_metrics['auc'])

        print(f"\nEpoch {epoch + 1} Summary:");
        print(f"  - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}");
        print("  - Validation Metrics:");
        [print(f"    - {k.capitalize()}: {v:.4f}") for k, v in val_metrics.items()]

        writer.add_scalar('Loss/train', train_loss, epoch);
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        for key, value in val_metrics.items(): writer.add_scalar(f'{key.capitalize()}/validation', value, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_metrics['loss'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            print(f"New best validation AUC: {best_val_auc:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(run_dir, "best_hybrid_model.pth"))

    writer.close()

    # --- 最终评估 ---
    print("\n--- Final Evaluation on Test Set ---")
    best_model = ECG_CNN_LSTM(n_leads=X_train.shape[2], n_classes=1).to(device)
    best_model.load_state_dict(torch.load(os.path.join(run_dir, "best_hybrid_model.pth")))
    test_metrics, test_labels, test_preds = evaluate(best_model, test_loader, criterion, device)

    print("Test Set Metrics:");
    [print(f"  - {k.capitalize()}: {v:.4f}") for k, v in test_metrics.items()]
    plot_metrics(history, save_path=os.path.join(run_dir, "training_curves_hybrid.png"))
    plot_confusion_matrix(test_labels, test_preds, class_names=['Class 0', 'Class 1'],
                          save_path=os.path.join(run_dir, "confusion_matrix_hybrid.png"))

    print(f"\nTraining complete! Best model and visualizations are saved in {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Hybrid model for ECG Classification")
    parser.add_argument("--data_dir", type=str, default="processed_data_split")
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    main(args)