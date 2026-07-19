"""
导出降维后的特征 CSV
对三个数据集分别采用 PCA 和 LDA 降维 + StandardScaler 标准化，输出带标签的 CSV 文件
用法: python export_features.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import load_dataset, create_patches, apply_dim_reduce, apply_lda

OUTPUT_DIR = "./results/csv"
DATASETS = ["IndianPines", "PaviaU", "Houston"]
METHODS = ["pca", "lda"]
PCA_COMPONENTS = 30
WINDOW_SIZE = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  高光谱特征导出 (CSV)")
print("  处理流程: .mat → patches → 降维(PCA/LDA) → StandardScaler → CSV")
print("=" * 60)

for dataset_name in DATASETS:
    print(f"\n{'=' * 60}")
    print(f"  数据集: {dataset_name}")
    print(f"{'=' * 60}")

    # ---- 1. 加载原始数据 + 切 patches ----
    data, gt, _ = load_dataset(dataset_name, "./data")
    patches, labels, num_classes, label_map, _ = create_patches(
        data, gt, window_size=WINDOW_SIZE
    )
    print(f"  Patches: {patches.shape} → {len(labels)} 个样本")

    for method in METHODS:
        print(f"\n  [{method.upper()}] 降维 + 标准化...")

        # ---- 2. 确定降维维度 ----
        if method == "lda":
            max_dim = num_classes - 1
            n_comp = min(PCA_COMPONENTS, max_dim)
        else:
            n_comp = PCA_COMPONENTS

        # ---- 3. 降维 (PCA 或 LDA) ----
        if method == "lda":
            # 对大样本数据集，随机采样 1 万条做 LDA，避免 OOM
            if len(labels) > 15000:
                print(f"    样本过多({len(labels)})，随机采样 12000 条拟合 LDA...")
                idx = np.random.choice(len(labels), 12000, replace=False)
                sample_patches = patches[idx]
                sample_labels = labels[idx]
            else:
                sample_patches = patches
                sample_labels = labels
            reduced = apply_lda(sample_patches, sample_labels, n_components=n_comp, pca_pre=30)

            # ---- 4. 取中心像素作为特征向量 ----
            center = WINDOW_SIZE // 2
            features = reduced[:, center, center, :]  # (N, k)

            # ---- 5. StandardScaler 标准化 ----
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            print(f"    标准化完成")

            # ---- 6. 构建 DataFrame 并保存（LDA 采样模式用 sample_labels） ----
            col_names = [f"f{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=col_names)
            df["label"] = sample_labels if len(labels) > 15000 else labels
        else:
            reduced = apply_dim_reduce(patches, labels, method=method, n_components=n_comp)
            # reduced: (N, H, W, k)

            # ---- 4. 取中心像素作为特征向量 ----
            center = WINDOW_SIZE // 2
            features = reduced[:, center, center, :]  # (N, k)

            # ---- 5. StandardScaler 标准化 ----
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            means = scaler.mean_
            stds = scaler.scale_
            print(f"    标准化: mean range [{means.min():.4f}, {means.max():.4f}], "
                  f"std range [{stds.min():.4f}, {stds.max():.4f}]")

            # ---- 6. 构建 DataFrame 并保存 ----
            col_names = [f"f{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=col_names)
            df["label"] = labels

        fname = f"{dataset_name}_{method}.csv"
        fpath = os.path.join(OUTPUT_DIR, fname)
        df.to_csv(fpath, index=False)
        print(f"  ✓ 已保存: {fpath}  (形状: {df.shape})")

print(f"\n{'=' * 60}")
print(f"  完成！CSV 文件保存在: {OUTPUT_DIR}/")
print(f"  每个 CSV 文件格式: f0, f1, ..., fk, label")
print(f"  所有特征已经 StandardScaler 标准化 (均值=0, 标准差=1)")
print(f"{'=' * 60}")