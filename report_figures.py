"""
生成报告用对比图：每个数据集一张大图，包含伪彩色原图、真值标签、模型预测
用法: python report_figures.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

from data_loader import load_dataset, create_patches, apply_pca, HyperSpectralDataset
from model import HybridSN
from sklearn.preprocessing import StandardScaler

# ---------- 配置 ----------
WINDOW_SIZE = 25
PCA_COMP = 30  # 默认 PCA 维度
DEVICE = torch.device("cpu")

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_FILES = {
    "IndianPines": os.path.join(RESULTS_DIR, "hybridsn_IndianPines.pth"),
    "PaviaU": os.path.join(RESULTS_DIR, "hybridsn_PaviaU.pth"),
    "Houston": os.path.join(RESULTS_DIR, "hybridsn_Houston.pth"),
}

# Houston 训练时 pca=20
HOUSTON_PCA = 20

# ==================== 颜色映射 ====================
# 使用高对比度离散色板
INDIAN_COLORS = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
                 '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080',
                 '#808000', '#800080', '#008080', '#C0C0C0', '#FFA500',
                 '#A52A2A', '#FFC0CB']
PAVIA_COLORS = ['#000000', '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c']
HOUSTON_COLORS = ['#000000', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
                  '#fb9a99', '#e31a1c', '#fdbf6f']


def load_model(path, num_classes, input_channels, patch_size):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", {})
    model = HybridSN(num_classes=num_classes,
                     input_channels=input_channels,
                     patch_size=patch_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"  加载模型: {os.path.basename(path)}, 保存时 OA={ckpt.get('test_acc','N/A')}%")
    return model


def predict_full(model, patches, labels, batch_size=32):
    """全量预测，返回 preds 数组和 OA 等指标"""
    # 标准化
    scaler = StandardScaler()
    flat = patches.reshape(len(patches), -1)
    scaler.fit(flat)
    patches_norm = scaler.transform(flat).reshape(patches.shape)

    dataset = HyperSpectralDataset(patches_norm, labels)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, batch_preds = torch.max(outputs, 1)
            all_preds.extend(batch_preds.cpu().numpy())
    all_preds = np.array(all_preds)

    from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
    acc = accuracy_score(labels, all_preds) * 100
    cm = confusion_matrix(labels, all_preds)
    kappa = cohen_kappa_score(labels, all_preds)
    per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    aa = per_class.mean() * 100
    return all_preds, acc, aa, kappa


def make_comparison_figure(data, gt, preds, positions, num_classes,
                           colors, title, save_path):
    """
    data: (H,W,C) 原始数据
    gt: (H,W) 真值标签
    preds: (N,) 预测标签 (0~num_classes-1)
    positions: (N,2) 每个 pred 对应的 (row,col)
    num_classes: 有效类别数（不含背景）
    colors: 颜色列表（索引0为背景黑）
    """
    h, w = gt.shape
    # 重建预测图
    pred_map = np.zeros_like(gt, dtype=np.int64)
    for idx, (r, c) in enumerate(positions):
        pred_map[r, c] = preds[idx] + 1  # preds 是 0-based，gt 中 1~N

    # 伪彩色合成
    bands = data.shape[-1]
    idxs = [bands//3, bands*2//3, 0]
    rgb = np.stack([data[:, :, idxs[0]], data[:, :, idxs[1]], data[:, :, idxs[2]]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    cmap = ListedColormap(colors[:num_classes+1])

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Pseudo-RGB Composite", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes, interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes, interpolation="nearest")
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    # 错误图
    mask = pred_map > 0
    correct = (gt == pred_map) & mask
    error_map = np.zeros((h, w, 3), dtype=np.float32)
    error_map[mask & correct] = [0, 1, 0]
    error_map[mask & ~correct] = [1, 0, 0]
    axes[3].imshow(error_map)
    axes[3].set_title("Correct (Green) vs Error (Red)", fontsize=12)
    axes[3].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: {save_path}")


def main():
    print("=" * 60)
    print("  生成报告用对比图")
    print("=" * 60)

    # ---- Indian Pines ----
    print("\n[Indian Pines]")
    data, gt, _ = load_dataset("IndianPines")
    patches, labels, num_classes, _, all_positions = create_patches(data, gt, WINDOW_SIZE)
    patches = apply_pca(patches, PCA_COMP)
    model = load_model(MODEL_FILES["IndianPines"], num_classes, PCA_COMP, WINDOW_SIZE)
    preds, acc, aa, kappa = predict_full(model, patches, labels)
    print(f"  OA={acc:.2f}%  AA={aa:.2f}%  Kappa={kappa:.4f}")
    make_comparison_figure(data, gt, preds, all_positions, num_classes,
                           INDIAN_COLORS,
                           f"Indian Pines (OA={acc:.2f}%, Kappa={kappa:.4f})",
                           os.path.join(FIG_DIR, "IndianPines_report.png"))

    # ---- PaviaU ----
    print("\n[PaviaU]")
    data, gt, _ = load_dataset("PaviaU")
    patches, labels, num_classes, _, all_positions = create_patches(data, gt, WINDOW_SIZE)
    patches = apply_pca(patches, PCA_COMP)
    model = load_model(MODEL_FILES["PaviaU"], num_classes, PCA_COMP, WINDOW_SIZE)
    preds, acc, aa, kappa = predict_full(model, patches, labels)
    print(f"  OA={acc:.2f}%  AA={aa:.2f}%  Kappa={kappa:.4f}")
    make_comparison_figure(data, gt, preds, all_positions, num_classes,
                           PAVIA_COLORS,
                           f"Pavia University (OA={acc:.2f}%, Kappa={kappa:.4f})",
                           os.path.join(FIG_DIR, "PaviaU_report.png"))

    # ---- Houston ----
    print("\n[Houston]")
    data, gt, _ = load_dataset("Houston")
    patches, labels, num_classes, _, all_positions = create_patches(data, gt, WINDOW_SIZE)
    patches = apply_pca(patches, HOUSTON_PCA)
    model = load_model(MODEL_FILES["Houston"], num_classes, HOUSTON_PCA, WINDOW_SIZE)
    preds, acc, aa, kappa = predict_full(model, patches, labels)
    print(f"  OA={acc:.2f}%  AA={aa:.2f}%  Kappa={kappa:.4f}")
    make_comparison_figure(data, gt, preds, all_positions, num_classes,
                           HOUSTON_COLORS,
                           f"Houston (OA={acc:.2f}%, Kappa={kappa:.4f})",
                           os.path.join(FIG_DIR, "Houston_report.png"))

    print("\n" + "=" * 60)
    print(f"  完成！图片保存在 {FIG_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()