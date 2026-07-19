"""
可视化模块
1. 数据可视化：伪彩色图、光谱曲线、标签分布
2. 分类结果可视化：训练曲线、混淆矩阵、分类结果图
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，适合服务器
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# ==================== 颜色配置 ====================

# Indian Pines 16 类地物的颜色映射
INDIAN_PINES_COLORS = [
    "#000000",  # 0 - 背景 (黑)
    "#FF0000",  # 1 - Alfalfa
    "#00FF00",  # 2 - Corn-notill
    "#0000FF",  # 3 - Corn-mintill
    "#FFFF00",  # 4 - Corn
    "#FF00FF",  # 5 - Grass-pasture
    "#00FFFF",  # 6 - Grass-trees
    "#800000",  # 7 - Grass-pasture-mowed
    "#008000",  # 8 - Hay-windrowed
    "#000080",  # 9 - Oats
    "#808000",  # 10 - Soybean-notill
    "#800080",  # 11 - Soybean-mintill
    "#008080",  # 12 - Soybean-clean
    "#C0C0C0",  # 13 - Wheat
    "#FFA500",  # 14 - Woods
    "#A52A2A",  # 15 - Buildings-Grass-Trees-Drives
    "#FFC0CB",  # 16 - Stone-Steel-Towers
]

INDIAN_PINES_NAMES = [
    "Background",
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
    "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
    "Soybean-clean", "Wheat", "Woods",
    "Buildings-Grass-Trees", "Stone-Steel-Towers",
]


# ==================== 数据可视化 ====================

def visualize_data(data, gt, save_dir="./results/figures"):
    """
    加载数据后的可视化：伪彩色图、光谱曲线、标签分布

    参数:
        data: (H, W, C) 高光谱数据
        gt:   (H, W) 标签图
        save_dir: 图片保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    num_classes = int(gt.max())

    # ---------- Fig 1: 伪彩色图 + 标签图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a. 伪彩色合成（RGB 用波段 29, 19, 9 近似真彩色）
    rgb = np.stack([
        data[:, :, 29],
        data[:, :, 19],
        data[:, :, 9],
    ], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    axes[0].imshow(rgb)
    axes[0].set_title("Pseudo-RGB (Bands 29-19-9)", fontsize=12)
    axes[0].axis("off")

    # 1b. PCA 第一主成分（灰度）
    from sklearn.decomposition import PCA
    data_flat = data.reshape(-1, data.shape[-1])
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(data_flat).reshape(data.shape[0], data.shape[1])
    axes[1].imshow(pc1, cmap="gray")
    axes[1].set_title("PCA First Component", fontsize=12)
    axes[1].axis("off")

    # 1c. 标签图
    cmap = ListedColormap(INDIAN_PINES_COLORS[:num_classes + 1])
    im = axes[2].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes, interpolation="nearest")
    axes[2].set_title("Ground Truth Labels", fontsize=12)
    axes[2].axis("off")

    # 图例
    patches = []
    for i in range(1, num_classes + 1):
        patches.append(mpatches.Patch(color=INDIAN_PINES_COLORS[i],
                                       label=INDIAN_PINES_NAMES[i]))
    axes[2].legend(handles=patches, loc="center left",
                   bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "01_data_overview.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 数据概览图已保存: 01_data_overview.png")

    # ---------- Fig 2: 光谱曲线 ----------
    fig, ax = plt.subplots(figsize=(12, 5))

    wavelengths = np.arange(data.shape[-1])
    for cls in range(1, num_classes + 1):
        mask = gt == cls
        if mask.sum() == 0:
            continue
        mean_spectrum = data[mask].mean(axis=0)
        ax.plot(wavelengths, mean_spectrum, label=INDIAN_PINES_NAMES[cls],
                linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Band Index", fontsize=12)
    ax.set_ylabel("Mean Reflectance", fontsize=12)
    ax.set_title("Mean Spectral Curves of All Classes", fontsize=14)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "02_spectral_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ 光谱曲线图已保存: 02_spectral_curves.png")

    # ---------- Fig 3: 类别分布（柱状图） ----------
    fig, ax = plt.subplots(figsize=(14, 5))
    counts = [(gt == i).sum() for i in range(1, num_classes + 1)]
    names = [f"{INDIAN_PINES_NAMES[i]}\n({i})" for i in range(1, num_classes + 1)]
    bars = ax.bar(range(len(counts)), counts, color=INDIAN_PINES_COLORS[1:num_classes + 1],
                  edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Labeled Pixels", fontsize=12)
    ax.set_title("Class Distribution (Labeled Pixels per Class)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "03_class_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ 类别分布图已保存: 03_class_distribution.png")

    # ---------- Fig 4: PCA 解释方差 ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    pca_full = PCA(n_components=min(50, data.shape[-1]))
    pca_full.fit(data_flat)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum) + 1), cumsum, "b-o", markersize=3)
    ax.axhline(y=0.99, color="r", linestyle="--", label="99% variance")
    ax.axvline(x=30, color="green", linestyle="--", label="30 components (used)")
    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("PCA Cumulative Explained Variance", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "04_pca_variance.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ PCA 方差图已保存: 04_pca_variance.png")


# ==================== 训练结果可视化 ====================

def visualize_training_curves(history, save_dir="./results/figures"):
    """
    训练曲线可视化：Loss 和 Accuracy

    参数:
        history: dict with "train_loss", "train_acc", "val_loss", "val_acc"
        save_dir: 图片保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 标注最佳 epoch
    best_epoch = np.argmax(history["val_acc"]) + 1
    best_acc = max(history["val_acc"])
    ax2.annotate(f"Best: Epoch {best_epoch}\nAcc={best_acc:.2f}%",
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + len(epochs) * 0.15, best_acc - 5),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=10, color="green", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "05_training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ 训练曲线图已保存: 05_training_curves.png")


def visualize_confusion_matrix(cm, num_classes, save_dir="./results/figures"):
    """
    混淆矩阵可视化

    参数:
        cm: (num_classes, num_classes) 混淆矩阵
        num_classes: 类别数
        save_dir: 图片保存目录
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs(save_dir, exist_ok=True)

    # 归一化混淆矩阵
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 原始数值 + 热力图
    im1 = ax1.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_title("Confusion Matrix (Normalized)", fontsize=14)
    ax1.set_xlabel("Predicted Class", fontsize=12)
    ax1.set_ylabel("True Class", fontsize=12)

    # 在格子中写数值
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax1.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                         fontsize=7, color=color)

    ax1.set_xticks(range(num_classes))
    ax1.set_yticks(range(num_classes))
    ax1.set_xticklabels(range(num_classes), fontsize=7, rotation=45)
    ax1.set_yticklabels(range(num_classes), fontsize=7)
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 每类准确率柱状图
    per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    colors = ["#2ca02c" if acc >= 0.9 else "#ff7f0e" if acc >= 0.7 else "#d62728"
              for acc in per_class]
    bars = ax2.bar(range(num_classes), per_class * 100, color=colors, edgecolor="black")
    ax2.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90%")
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels(range(num_classes), fontsize=8)
    ax2.set_xlabel("Class Index", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Per-Class Accuracy", fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    for bar, acc in zip(bars, per_class):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc * 100:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "06_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ 混淆矩阵图已保存: 06_confusion_matrix.png")


def visualize_classification_map(gt, preds, positions, save_dir="./results/figures",
                                 num_classes=16):
    """
    分类结果图：对比标签图和预测图

    参数:
        gt:           (H, W) 原始标签（含背景 0）
        preds:        (N,) 预测标签（0~num_classes-1），若长度<positions则只重建有pred的部分
        positions:    (N, 2) 每个预测对应的 (row, col)
        save_dir:     保存目录
        num_classes:  有效类别数（不含背景）
    """
    os.makedirs(save_dir, exist_ok=True)

    h, w = gt.shape

    # 重建预测图（背景填 0）
    pred_map = np.zeros_like(gt, dtype=np.int64)
    n_pred = min(len(preds), len(positions))
    for idx in range(n_pred):
        r, c = positions[idx]
        pred_map[r, c] = preds[idx] + 1  # +1 因为 preds 从 0 开始，gt 中 1~16

    # 只对比有预测的像素
    mask = pred_map > 0
    correct = (gt == pred_map) & mask

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    cmap = ListedColormap(INDIAN_PINES_COLORS[:num_classes + 1])

    # a. 真值标签
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes, interpolation="nearest")
    axes[0].set_title("Ground Truth", fontsize=12)
    axes[0].axis("off")

    # b. 预测结果
    axes[1].imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes, interpolation="nearest")
    axes[1].set_title("Prediction", fontsize=12)
    axes[1].axis("off")

    # c. 错误图（红色=错误，绿色=正确，黑色=背景）
    error_map = np.zeros((h, w, 3), dtype=np.float32)
    error_map[mask & correct] = [0, 1, 0]          # 正确 → 绿色
    error_map[mask & ~correct] = [1, 0, 0]         # 错误 → 红色
    axes[2].imshow(error_map)
    axes[2].set_title("Correct (Green) vs Error (Red)", fontsize=12)
    axes[2].axis("off")

    # d. 图例
    patches = []
    for i in range(1, num_classes + 1):
        patches.append(mpatches.Patch(color=INDIAN_PINES_COLORS[i],
                                       label=f"{i}: {INDIAN_PINES_NAMES[i]}"))
    axes[3].legend(handles=patches, loc="center", fontsize=7, ncol=1)
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "07_classification_map.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 分类结果图已保存: 07_classification_map.png")


# ==================== 全流程可视化（供 train.py 调用） ====================

def plot_data_overview(data, gt, save_dir="./results/figures"):
    """供 train.py 在数据加载后调用"""
    visualize_data(data, gt, save_dir)


def plot_training_results(history, cm, preds, gt_full, positions, num_classes,
                          save_dir="./results/figures"):
    """供 train.py 在训练结束后调用"""
    visualize_training_curves(history, save_dir)
    visualize_confusion_matrix(cm, num_classes, save_dir)
    visualize_classification_map(gt_full, preds, positions, save_dir, num_classes)


# ==================== 独立测试 ====================

if __name__ == "__main__":
    print("=" * 55)
    print("  可视化模块测试")
    print("=" * 55)

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_dataset

    # --- 测试数据可视化 ---
    print("\n[1] 数据可视化测试")
    print("-" * 40)
    data, gt, _ = load_dataset("IndianPines", "./data")
    visualize_data(data, gt, "./results/figures/test")

    # --- 测试训练曲线 ---
    print("\n[2] 训练曲线测试")
    print("-" * 40)
    dummy_history = {
        "train_loss": [2.0, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03],
        "val_loss": [2.2, 1.2, 0.6, 0.4, 0.25, 0.18, 0.12, 0.09, 0.06, 0.04],
        "train_acc": [30, 50, 65, 75, 82, 88, 92, 95, 97, 98],
        "val_acc": [28, 48, 62, 73, 80, 85, 90, 93, 96, 97.5],
    }
    visualize_training_curves(dummy_history, "./results/figures/test")

    # --- 测试混淆矩阵 ---
    print("\n[3] 混淆矩阵测试")
    print("-" * 40)
    np.random.seed(42)
    cm_dummy = np.zeros((16, 16), dtype=int)
    for i in range(16):
        n = np.random.randint(20, 500)
        probs = np.random.dirichlet(np.ones(16) * 0.1)
        probs[i] *= 20
        probs /= probs.sum()
        cm_dummy[i] = np.random.multinomial(n, probs)
    visualize_confusion_matrix(cm_dummy, 16, "./results/figures/test")

    print("\n" + "=" * 55)
    print("  可视化测试完成！图片保存在 ./results/figures/test/")
    print("=" * 55)