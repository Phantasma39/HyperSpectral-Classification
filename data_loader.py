"""
高光谱数据加载器
支持 Indian Pines, Pavia University, Houston 等数据集
"""
import os
import ssl
import zipfile

# 全局禁用 SSL 证书验证（学校网络环境兼容）
ssl._create_default_https_context = ssl._create_unverified_context

import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


# ==================== 数据集下载 ====================

DATASET_URLS = {
    "IndianPines": {
        "url": "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines_corrected.mat",
        "data_key": "indian_pines_corrected",
        "gt_url": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        "gt_key": "indian_pines_gt",
    },
    "PaviaU": {
        "url": "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "data_key": "paviaU",
        "gt_url": "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "gt_key": "paviaU_gt",
    },
    "Houston": {
        "url": "http://www.ehu.eus/ccwintco/uploads/9/93/Classification_Matlab_Mat.zip",
        "data_key": "houston",
        "gt_url": "",
        "gt_key": "houston_gt",
    },
}


# 数据文件可能的存放路径（自动搜索）
DATA_SEARCH_DIRS = [
    "./data",
    "./HSI-SVM-master/Indian Pines",
    "./HSI-SVM-master/data",
    "../Indian Pines",
]


def find_data_file(filename):
    """在多个可能路径中查找数据文件"""
    for d in DATA_SEARCH_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def download_file(url, save_path):
    """检查文件是否存在，不存在则给出手动下载指引"""
    # 先尝试原始路径
    if os.path.exists(save_path):
        print(f"  文件已存在: {save_path}")
        return

    # 再尝试搜索其他路径
    found = find_data_file(os.path.basename(save_path))
    if found:
        print(f"  文件已找到: {found}")
        # 拷贝到目标位置
        import shutil
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if found != os.path.abspath(save_path):
            shutil.copy2(found, save_path)
            print(f"  已复制到: {save_path}")
        return

    print(f"\n  ⚠ 文件缺失: {save_path}")
    print(f"  请在以下链接手动下载，放到 ./data/ 目录下:")
    print(f"    {url}")
    raise FileNotFoundError(f"请先手动下载数据文件: {save_path}")


def load_dataset(name="IndianPines", data_dir="./data"):
    """
    加载高光谱数据集

    参数:
        name: 数据集名称 ("IndianPines", "PaviaU", "Houston")
        data_dir: 数据存放目录

    返回:
        data: (H, W, C) 高光谱数据
        gt:   (H, W) 标签图
        num_classes: 类别数（含背景0类）
    """
    if name not in DATASET_URLS:
        raise ValueError(f"不支持的数据集: {name}，可选: {list(DATASET_URLS.keys())}")

    info = DATASET_URLS[name]
    os.makedirs(data_dir, exist_ok=True)

    # 下载数据文件
    data_path = os.path.join(data_dir, os.path.basename(info["url"]))
    download_file(info["url"], data_path)

    # 下载标签文件
    if name != "Houston":
        gt_path = os.path.join(data_dir, os.path.basename(info["gt_url"]))
        download_file(info["gt_url"], gt_path)
        data = sio.loadmat(data_path)[info["data_key"]]
        gt = sio.loadmat(gt_path)[info["gt_key"]]
    else:
        # Houston 需要解压
        if not os.path.exists(os.path.join(data_dir, "Houston.mat")):
            with zipfile.ZipFile(data_path, 'r') as zf:
                zf.extractall(data_dir)
        houston_data = sio.loadmat(os.path.join(data_dir, "Houston.mat"))
        data = houston_data["Houston"]
        gt = houston_data["Houston_gt"]

    data = data.astype(np.float32)
    gt = gt.astype(np.int64).squeeze()

    num_classes = len(np.unique(gt))  # 含 0（背景）

    print(f"  数据集: {name}")
    print(f"  数据形状: {data.shape} (H×W×C)")
    print(f"  标签形状: {gt.shape}")
    print(f"  类别数（含背景）: {num_classes}")
    print(f"  有效类别数（不含背景）: {num_classes - 1}")

    return data, gt, num_classes


# ==================== 数据预处理 ====================

def pad_with_zeros(data, margin):
    """用零填充数据边界，方便取邻近像素做 patch"""
    new_data = np.zeros(
        (data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2]),
        dtype=data.dtype,
    )
    new_data[margin: margin + data.shape[0], margin: margin + data.shape[1], :] = data
    return new_data


def create_patches(data, gt, window_size=25, remove_zero_labels=True):
    """
    以每个标注像素为中心，提取 window_size × window_size 的 patch

    参数:
        data:        (H, W, C) 数据
        gt:          (H, W) 标签
        window_size: patch 边长
        remove_zero_labels: 是否过滤标签为 0（背景）的像素

    返回:
        patches: (N, window_size, window_size, C)
        labels:  (N,) 从 0 开始编号（移除了背景后重新编号）
        positions: (N, 2) (row, col) 原始位置
    """
    margin = (window_size - 1) // 2
    padded_data = pad_with_zeros(data, margin)

    patches = []
    labels = []
    positions = []

    h, w = gt.shape
    for i in range(h):
        for j in range(w):
            label = gt[i, j]
            if remove_zero_labels and label == 0:
                continue
            patch = padded_data[i: i + window_size, j: j + window_size, :]
            patches.append(patch)
            labels.append(label)
            positions.append((i, j))

    patches = np.array(patches, dtype=np.float32)
    labels_orig = np.array(labels, dtype=np.int64)
    positions = np.array(positions)

    # 将标签映射为 0 ~ num_classes-1
    unique_labels = np.unique(labels_orig)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels_new = np.array([label_map[l] for l in labels_orig], dtype=np.int64)

    num_classes = len(unique_labels)

    print(f"  提取 patches: {patches.shape}")
    print(f"  类别分布:")
    for old_lbl, new_lbl in label_map.items():
        count = np.sum(labels_new == new_lbl)
        print(f"    类别 {old_lbl} → {new_lbl}: {count} 个样本")

    return patches, labels_new, num_classes, label_map, positions


def apply_pca(data, n_components=30):
    """
    对最后一个维度（光谱波段）应用 PCA 降维

    参数:
        data: (N, H, W, C) 或 (H, W, C) 或 (N, C)
        n_components: 保留的主成分数

    返回:
        降维后的数据
    """
    shape = data.shape
    if data.ndim == 4:
        N, H, W, C = shape
        data_2d = data.reshape(-1, C)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(data_2d)
        print(f"  PCA: {C} → {n_components} 波段，解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
        return reduced.reshape(N, H, W, n_components)
    elif data.ndim == 3:
        H, W, C = shape
        data_2d = data.reshape(-1, C)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(data_2d)
        print(f"  PCA: {C} → {n_components} 波段，解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
        return reduced.reshape(H, W, n_components)
    else:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(data)
        print(f"  PCA: {data.shape[1]} → {n_components} 波段，解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
        return reduced


# ==================== PyTorch Dataset ====================

class HyperSpectralDataset(Dataset):
    """高光谱 PyTorch 数据集"""

    def __init__(self, patches, labels, transform=None):
        self.patches = patches  # (N, H, W, C)
        self.labels = labels    # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.patches[idx]  # (H, W, C)
        label = self.labels[idx]

        # 转换为 (C, H, W) 适合 PyTorch 卷积输入
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            patch = self.transform(patch)

        return patch, label


def create_data_loaders(patches, labels, train_ratio=0.7, val_ratio=0.1,
                        batch_size=32, random_state=42, num_workers=0):
    """
    划分训练/验证/测试集并创建 DataLoader

    参数:
        patches:      (N, H, W, C)
        labels:       (N,)
        train_ratio:  训练集比例
        val_ratio:    验证集比例
        batch_size:   批大小
        random_state: 随机种子
        num_workers:  数据加载线程数

    返回:
        train_loader, val_loader, test_loader, class_weights
    """
    # 计算类别权重（用于平衡损失）
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum() * len(unique)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # 分层划分
    X_temp, X_test, y_temp, y_test = train_test_split(
        patches, labels, test_size=1 - train_ratio - val_ratio,
        stratify=labels, random_state=random_state
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted,
        stratify=y_temp, random_state=random_state
    )

    print(f"\n  数据集划分:")
    print(f"    训练集: {len(X_train)} 样本")
    print(f"    验证集: {len(X_val)} 样本")
    print(f"    测试集: {len(X_test)} 样本")

    # 标准化：基于训练集
    scaler = StandardScaler()
    train_flat = X_train.reshape(len(X_train), -1)
    scaler.fit(train_flat)

    X_train = (scaler.transform(X_train.reshape(len(X_train), -1))
               .reshape(X_train.shape))
    X_val = (scaler.transform(X_val.reshape(len(X_val), -1))
             .reshape(X_val.shape))
    X_test = (scaler.transform(X_test.reshape(len(X_test), -1))
              .reshape(X_test.shape))

    train_dataset = HyperSpectralDataset(X_train, y_train)
    val_dataset = HyperSpectralDataset(X_val, y_val)
    test_dataset = HyperSpectralDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights, scaler


# ==================== 快速测试 ====================

if __name__ == "__main__":
    print("=" * 55)
    print("  数据加载器测试")
    print("=" * 55)
    print()

    # 测试 Indian Pines
    data, gt, num_classes = load_dataset("IndianPines", "./data")

    patches, labels, n_cls, label_map = create_patches(data, gt, window_size=25)

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        patches, labels, train_ratio=0.7, val_ratio=0.1, batch_size=32
    )

    # 取出一个 batch 测试
    batch_x, batch_y = next(iter(train_loader))
    print(f"\n  测试 batch: x {list(batch_x.shape)}, y {list(batch_y.shape)}")
    print(f"  类别权重: {class_weights.numpy()}")

    print("\n  数据加载器测试通过！✓")