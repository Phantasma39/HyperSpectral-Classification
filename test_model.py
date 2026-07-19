"""
测试脚本：加载已训练模型，随机测试并可视化
用法: python test_model.py --model_path results/hybridsn_IndianPines.pth --num_samples 10
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from data_loader import (
    load_dataset, create_patches, apply_pca, HyperSpectralDataset,
    create_data_loaders,
)
from model import HybridSN, HybridSN_SE, HybridSN_Res
from visualize import (
    INDIAN_PINES_COLORS, INDIAN_PINES_NAMES,
    plot_data_overview, plot_training_results,
)


# ==================== 核心测试逻辑 ====================

def load_trained_model(model_path, device):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    print(f"\n  加载模型: {model_path}")
    print(f"  配置: {config}")
    print(f"  保存时的测试准确率: {checkpoint.get('test_acc', 'N/A')}%")
    print(f"  保存时的 Kappa: {checkpoint.get('kappa', 'N/A')}")

    # 从文件名推断模型类型
    if "hybridsn_res" in os.path.basename(model_path).lower():
        model = HybridSN_Res(**config)
    elif "hybridsn_se" in os.path.basename(model_path).lower():
        model = HybridSN_SE(**config)
    else:
        model = HybridSN(**config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    label_map = checkpoint.get("label_map", None)

    return model, config, label_map


def random_test_samples(model, patches, labels, all_positions, gt_full,
                        num_samples=10, device="cpu"):
    """
    随机抽取 num_samples 个样本测试，返回结果

    返回:
        results: list of dict
    """
    n = len(labels)
    indices = random.sample(range(n), min(num_samples, n))

    results = []
    for idx in indices:
        patch = torch.from_numpy(patches[idx]).permute(2, 0, 1).unsqueeze(0).float().to(device)
        true_label = labels[idx]
        row, col = all_positions[idx]
        gt_original = gt_full[row, col]  # 原始标签（含背景 0）

        with torch.no_grad():
            output = model(patch)
            probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
            pred_label = int(torch.argmax(output, dim=1).item())
            confidence = float(probs[pred_label] * 100)

        results.append({
            "idx": idx,
            "row": int(row),
            "col": int(col),
            "true_label": int(true_label),
            "gt_original": int(gt_original),
            "pred_label": int(pred_label),
            "confidence": confidence,
            "correct": true_label == pred_label,
            "probs": probs,
            "patch": patches[idx],  # (H, W, C) numpy array
        })

    return results


def print_test_results(results, num_classes):
    """打印测试结果表格"""
    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)

    print(f"\n{'='*70}")
    print(f"  随机测试结果 ({correct_count}/{total} 正确, {correct_count/total*100:.1f}%)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'位置':<12} {'真实':<8} {'预测':<8} {'置信度(%)':<12} {'结果':<6}")
    print(f"{'-'*70}")

    for i, r in enumerate(results):
        result_str = "✓" if r["correct"] else "✗"
        print(f"{i+1:<4} ({r['row']:>3},{r['col']:>3})   "
              f"{r['true_label']:<8} {r['pred_label']:<8} "
              f"{r['confidence']:<12.1f} {result_str:<6}")

    # 统计各类别准确率
    print(f"\n  各类别统计:")
    for cls in range(num_classes):
        cls_results = [r for r in results if r["true_label"] == cls]
        if cls_results:
            cls_correct = sum(1 for r in cls_results if r["correct"])
            print(f"    类别 {cls}: {cls_correct}/{len(cls_results)} ({cls_correct/len(cls_results)*100:.1f}%)")


def visualize_test_samples(results, save_dir="./results/figures/test"):
    """
    可视化随机测试样本：每个样本展示 patch 图和预测信息
    """
    os.makedirs(save_dir, exist_ok=True)

    n = len(results)
    if n == 0:
        return

    # 每行最多 5 个
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, r in enumerate(results):
        row, col_idx = i // cols, i % cols
        ax = axes[row, col_idx]

        # 显示 PCA 降维后的 patch 的合成伪彩色
        patch = r["patch"]  # (H, W, C)
        if patch.shape[-1] >= 3:
            rgb = np.stack([
                patch[:, :, patch.shape[-1] // 3],
                patch[:, :, patch.shape[-1] * 2 // 3],
                patch[:, :, 0],
            ], axis=-1)
        else:
            rgb = patch[:, :, :3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        ax.imshow(rgb)

        color = "green" if r["correct"] else "red"
        ax.set_title(f"True: {r['gt_original']}\nPred: {r['pred_label']} ({r['confidence']:.0f}%)",
                     color=color, fontsize=9)
        ax.axis("off")

    # 隐藏多余子图
    for i in range(n, rows * cols):
        row, col_idx = i // cols, i % cols
        axes[row, col_idx].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "test_samples.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 测试样本图已保存: {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 1. 载入模型 ----
    print("=" * 55)
    print("  1. 加载模型")
    print("=" * 55)
    model, config, label_map = load_trained_model(args.model_path, device)

    # ---- 2. 加载数据 ----
    print("\n" + "=" * 55)
    print("  2. 加载数据")
    print("=" * 55)
    data, gt, _ = load_dataset(args.dataset, args.data_dir)

    # ---- 数据可视化概览 ----
    fig_dir = os.path.join(args.output_dir, "figures")
    plot_data_overview(data, gt, fig_dir)

    patches, labels, num_classes, label_map2, all_positions = create_patches(
        data, gt, window_size=args.window_size
    )

    # PCA
    if args.pca > 0:
        patches = apply_pca(patches, n_components=args.pca)

    # 标准化（兼容训练时的 scaler）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    flat = patches.reshape(len(patches), -1)
    scaler.fit(flat)
    patches = scaler.transform(flat).reshape(patches.shape)

    print(f"  样本总数: {len(labels)}")

    # ---- 3. 全量测试（评估） ----
    print("\n" + "=" * 55)
    print("  3. 全量测试评估")
    print("=" * 55)

    from torch.utils.data import DataLoader
    dataset = HyperSpectralDataset(patches, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
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

    print(f"\n  全量 Overall Accuracy (OA): {acc:.2f}%")
    print(f"  全量 Average Accuracy (AA):  {aa:.2f}%")
    print(f"  全量 Kappa 系数:              {kappa:.4f}")

    # ---- 4. 随机抽样测试 ----
    print("\n" + "=" * 55)
    print("  4. 随机抽样测试")
    print("=" * 55)

    results = random_test_samples(
        model, patches, labels, all_positions, gt,
        num_samples=args.num_samples, device=device
    )
    print_test_results(results, num_classes)

    # ---- 5. 可视化 ----
    print("\n" + "=" * 55)
    print("  5. 生成可视化")
    print("=" * 55)

    # 随机样本图
    visualize_test_samples(results, os.path.join(args.output_dir, "figures"))

    # 混淆矩阵 + 分类结果图（跳过训练曲线因为无 history）
    from visualize import visualize_confusion_matrix, visualize_classification_map
    visualize_confusion_matrix(cm, num_classes, os.path.join(args.output_dir, "figures"))
    visualize_classification_map(gt, all_preds, all_positions,
                                 os.path.join(args.output_dir, "figures"), num_classes)

    # ---- 6. 单张详细信息 ----
    print("\n" + "=" * 55)
    print("  6. 总结")
    print("=" * 55)
    print(f"\n  模型: {os.path.basename(args.model_path)}")
    print(f"  数据集: {args.dataset}")
    print(f"  全量 OA: {acc:.2f}%")
    print(f"  全量 AA: {aa:.2f}%")
    print(f"  全量 Kappa: {kappa:.4f}")
    print(f"  随机 {args.num_samples} 样本准确率: {sum(1 for r in results if r['correct'])/len(results)*100:.1f}%")
    print(f"\n  可视化文件保存在: {os.path.join(args.output_dir, 'figures')}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高光谱模型测试脚本")

    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的 .pth 模型路径")
    parser.add_argument("--dataset", type=str, default="IndianPines",
                        choices=["IndianPines", "PaviaU", "Houston"],
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据存放目录")
    parser.add_argument("--pca", type=int, default=30,
                        help="PCA 降维波段数（需与训练时一致）")
    parser.add_argument("--window_size", type=int, default=25,
                        help="Patch 大小（需与训练时一致）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=10,
                        help="随机抽取多少个样本测试")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="结果输出目录")

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main(args)