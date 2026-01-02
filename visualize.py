# visualize.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
import os


class MetricsVisualizer:
    def __init__(self, save_dir='plots'):
        """
        初始化可视化工具。
        Args:
            save_dir (str): 保存图表的目录。
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 初始化用于存储历史记录的列表
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }

        # 用于存储ROC曲线所需的数据
        self.y_true_val = []
        self.y_pred_probs_val = []

    def update(self, epoch_metrics: dict):
        """
        在每个epoch结束后，更新历史记录。
        Args:
            epoch_metrics (dict): 包含当前epoch各项指标的字典。
        """
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
        # 新增方法1: 保存ROC数据

    def update_roc_data(self, y_true, y_probs):
        self.roc_data = {'y_true': y_true, 'y_probs': y_probs}

        # 新增方法2: 加载历史记录

    def load_history(self, history_dict):
        self.history = history_dict
        print("Metrics history loaded.")

    def update_roc_data(self, y_true, y_pred_probs):
        """在验证循环中，收集用于绘制ROC曲线的真实标签和预测概率。"""
        self.y_true_val.extend(y_true)
        self.y_pred_probs_val.extend(y_pred_probs)

    def plot_all(self):
        """在训练结束后，绘制并保存所有图表。"""
        print("\n开始生成可视化图表...")
        self._plot_loss()
        self._plot_accuracy()
        self._plot_f1_score()
        self._plot_roc_curve()
        print(f"所有图表已保存到 '{self.save_dir}' 目录。")

    def _plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'))
        plt.close()

    def _plot_accuracy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'accuracy_curve.png'))
        plt.close()

    def _plot_f1_score(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_f1'], label='Train F1 Score')
        plt.plot(self.history['val_f1'], label='Validation F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'f1_score_curve.png'))
        plt.close()

    def _plot_roc_curve(self):
        if not self.y_true_val or not self.y_pred_probs_val:
            print("警告: 没有足够的ROC数据来绘图。")
            return

        fpr, tpr, _ = roc_curve(self.y_true_val, self.y_pred_probs_val)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'roc_curve.png'))
        plt.close()