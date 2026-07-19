"""
高光谱图像分类训练脚本
支持 HybridSN 及改进模型，可切换不同数据集
"""
import os
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, confusion_matrix, cohen_kappa_score,
    classification_report
)

from data_loader import load_dataset, create_patches, apply_pca, create_data_loaders, HyperSpectralDataset
from model import HybridSN, HybridSN_SE, HybridSN_Res
from visualize import plot_data_overview, plot_training_results


# ==================== 训练函数 ====================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估（验证/测试）"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, cm, kappa, all_preds, all_labels


def compute_per_class_accuracy(cm):
    """从混淆矩阵计算每一类的准确率"""
    per_class = cm.diagonal() / cm.sum(axis=1)
    per_class = np.nan_to_num(per_class, nan=0.0)
    return per_class


# ==================== 主训练流程 ====================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ========== 1. 加载数据 ==========
    print("\n" + "=" * 55)
    print("  1. 加载数据")
    print("=" * 55)

    data, gt, _ = load_dataset(args.dataset, args.data_dir)

    # ---- 数据可视化 ----
    fig_dir = os.path.join(args.output_dir, "figures")
    plot_data_overview(data, gt, fig_dir)

    patches, labels, num_classes, label_map, all_positions = create_patches(
        data, gt, window_size=args.window_size
    )

    # PCA 降维
    if args.pca > 0:
        patches = apply_pca(patches, n_components=args.pca)

    # 划分数据集
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        patches, labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        random_state=args.seed,
    )

    # ========== 2. 创建模型 ==========
    print("\n" + "=" * 55)
    print("  2. 创建模型")
    print("=" * 55)

    model_config = {
        "num_classes": num_classes,
        "input_channels": args.pca if args.pca > 0 else data.shape[-1],
        "patch_size": args.window_size,
        "dropout": args.dropout,
    }

    MODEL_MAP = {
        "hybridsn": HybridSN,
        "hybridsn_se": HybridSN_SE,
        "hybridsn_res": HybridSN_Res,
    }

    if args.model not in MODEL_MAP:
        raise ValueError(f"未知模型: {args.model}，可选: {list(MODEL_MAP.keys())}")

    model = MODEL_MAP[args.model](**model_config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  模型: {args.model}")
    print(f"  参数量: {params:,}")
    print(f"  配置: {model_config}")

    # ========== 3. 训练 ==========
    print("\n" + "=" * 55)
    print("  3. 训练")
    print("=" * 55)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Epoch':<7} {'Train Loss':<12} {'Train Acc':<10} "
          f"{'Val Loss':<12} {'Val Acc':<10} {'Time':<8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start

        print(f"{epoch:<7} {train_loss:<12.4f} {train_acc:<10.2f} "
              f"{val_loss:<12.4f} {val_acc:<10.2f} {elapsed:<8.1f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n  最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # ========== 4. 测试 ==========
    print("\n" + "=" * 55)
    print("  4. 测试")
    print("=" * 55)

    model.load_state_dict(best_model_wts)
    test_loss, test_acc, cm, kappa, preds, true_labels = evaluate(
        model, test_loader, criterion, device
    )

    per_class_acc = compute_per_class_accuracy(cm)

    print(f"\n  测试集 Overall Accuracy (OA): {test_acc:.2f}%")
    print(f"  测试集 Average Accuracy (AA):  {per_class_acc.mean() * 100:.2f}%")
    print(f"  测试集 Kappa 系数:              {kappa:.4f}")
    print(f"  测试集 Loss:                    {test_loss:.4f}")

    print(f"\n  每类准确率:")
    print(f"  {'类别':<6} {'原始标签':<8} {'准确率':<10} {'样本数':<8}")
    print(f"  {'-' * 35}")
    for cls_idx in range(num_classes):
        nb = cm.sum(axis=1)[cls_idx]
        print(f"  {cls_idx:<6} {cls_idx + 1:<8} {per_class_acc[cls_idx] * 100:<10.2f} {int(nb):<8}")

    # ========== 5. 保存结果 ==========
    print("\n" + "=" * 55)
    print("  5. 保存结果")
    print("=" * 55)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 全量预测（用于完整分类结果图） ----
    print("\n  生成完整分类结果图...")
    from torch.utils.data import DataLoader
    full_dataset = HyperSpectralDataset(patches, labels)  # 全量标准化后的数据
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    full_preds = []
    with torch.no_grad():
        for inputs, _ in full_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, batch_preds = torch.max(outputs, 1)
            full_preds.extend(batch_preds.cpu().numpy())
    full_preds = np.array(full_preds)

    # ---- 训练结果可视化 ----
    fig_dir = os.path.join(args.output_dir, "figures")
    plot_training_results(history, cm, full_preds, gt, all_positions, num_classes,
                          fig_dir)

    # 保存模型
    model_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}.pth")
    torch.save({
        "model_state_dict": best_model_wts,
        "config": model_config,
        "label_map": label_map,
        "test_acc": test_acc,
        "kappa": kappa,
    }, model_path)
    print(f"  模型已保存: {model_path}")

    # 保存分类报告
    report = classification_report(
        true_labels, preds,
        target_names=[f"Class_{i}" for i in range(num_classes)],
        digits=4,
    )

    report_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"模型: {args.model}\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"PCA 波段数: {args.pca}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"=" * 55 + "\n")
        f.write(f"Overall Accuracy (OA): {test_acc:.4f}%\n")
        f.write(f"Average Accuracy (AA):  {per_class_acc.mean() * 100:.4f}%\n")
        f.write(f"Kappa:                  {kappa:.4f}\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
        f.write("\n\n混淆矩阵:\n")
        f.write(str(cm))
    print(f"  报告已保存: {report_path}")

    return test_acc, kappa


# ==================== 命令行入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高光谱图像分类训练")

    # 数据
    parser.add_argument("--dataset", type=str, default="IndianPines",
                        choices=["IndianPines", "PaviaU", "Houston"],
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据存放目录")
    parser.add_argument("--window_size", type=int, default=25,
                        help="Patch 大小")
    parser.add_argument("--pca", type=int, default=30,
                        help="PCA 降维波段数（0 表示不降维）")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")

    # 模型
    parser.add_argument("--model", type=str, default="hybridsn",
                        choices=["hybridsn", "hybridsn_se", "hybridsn_res"],
                        help="模型名称")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout 比例")

    # 训练
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="结果输出目录")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 55)
    print("  高光谱图像分类训练")
    print("=" * 55)
    print(f"  参数: {vars(args)}")

    test_acc, kappa = main(args)

    print("\n" + "=" * 55)
    print(f"  最终结果: OA={test_acc:.2f}%, Kappa={kappa:.4f}")
    print("=" * 55)