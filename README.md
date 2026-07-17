# 🌈 HyperSpectral-Classification

高光谱图像智能解译大作业 — 基于 HybridSN 及其改进模型的高光谱图像分类

---

## 📖 项目简介

本仓库实现了**高光谱图像分类**的完整流程，包括数据预处理、PCA 降维、深度学习模型训练与评估。以 **HybridSN**（3D-2D CNN 混合架构）为基线，进一步实现了带通道注意力（SE）和残差连接（Residual）的改进版本。

支持的经典高光谱数据集：

| 数据集 | 空间尺寸 | 波段数 | 类别数 | 场景 |
|--------|---------|--------|--------|------|
| Indian Pines | 145 × 145 | 220 | 16 | 农田/森林 |
| Pavia University | 610 × 340 | 103 | 9 | 城市遥感 |
| Houston | 349 × 1905 | 144 | 15 | 城市遥感 |

---

## 🏗️ 模型架构

### 1. HybridSN（60 分基线）

原始论文结构，3 层 3D 卷积提取光谱-空间联合特征，reshape 后用 2D 卷积提取空间特征：

```
Input (B, C, H, W)
 → unsqueeze → (B, 1, C, H, W)
 → 3D Conv (kernel=7×3×3) + BN + ReLU → 8 ch
 → 3D Conv (kernel=5×3×3) + BN + ReLU → 16 ch
 → 3D Conv (kernel=3×3×3) + BN + ReLU → 32 ch
 → reshape → (B, 32×C', H', W')
 → 2D Conv + BN + ReLU → 64 ch
 → Flatten → FC(256) → FC(128) → SoftMax
```

### 2. HybridSN-SE（80 分改进）

在原始 HybridSN 的三层 3D 卷积和一层 2D 卷积后均加入 **Squeeze-and-Excitation** 通道注意力模块，自动学习各通道的重要性权重。

### 3. HybridSN-Res（90+ 分改进）

将 3D 卷积替换为含 SE 注意力的**残差块**（Residual Block），缓解深层网络的梯度退化，增强特征提取能力。

---

## 📦 安装

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn
```

验证环境：

```bash
python check_env.py
```

---

## 🚀 使用方法

### 基础训练（60 分）

```bash
python train.py --model hybridsn --dataset IndianPines --epochs 100
```

### 改进模型（80/90+ 分）

```bash
# 加 SE 注意力
python train.py --model hybridsn_se --dataset IndianPines --epochs 100

# 加残差 + SE（推荐）
python train.py --model hybridsn_res --dataset IndianPines --epochs 100
```

### 跨数据集对比

```bash
python train.py --model hybridsn_res --dataset PaviaU --epochs 100
```

### 自定义参数

```bash
python train.py \
  --model hybridsn_res \
  --dataset IndianPines \
  --pca 30 \
  --window_size 25 \
  --epochs 150 \
  --batch_size 64 \
  --lr 1e-3 \
  --dropout 0.4
```

**完整参数列表**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | IndianPines | 数据集 (IndianPines / PaviaU / Houston) |
| `--model` | hybridsn | 模型 (hybridsn / hybridsn_se / hybridsn_res) |
| `--pca` | 30 | PCA 降维波段数 |
| `--window_size` | 25 | 输入 patch 大小 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--dropout` | 0.4 | Dropout 比例 |
| `--train_ratio` | 0.7 | 训练集比例 |
| `--seed` | 42 | 随机种子 |

---

## 📊 评估指标

训练结束自动输出：

- **Overall Accuracy (OA)**：总正确率
- **Average Accuracy (AA)**：各类别平均准确率
- **Kappa 系数**：一致性系数（衡量分类与随机的一致性偏差）
- **Per-Class Accuracy**：每一类地物的分类精度
- **混淆矩阵**：直观可视化分类错误分布

---

## 📁 项目结构

```
├── data_loader.py   # 数据下载、PCA 降维、PyTorch Dataset/DataLoader
├── model.py         # HybridSN / HybridSN-SE / HybridSN-Res 模型定义
├── train.py         # 训练、验证、测试、评估全流程
├── check_env.py     # 环境依赖检查
├── .gitignore
└── README.md
```

---

## 🔗 参考

- [HybridSN 论文](https://ieeexplore.ieee.org/document/8736016) — Exploring 3-D–2-D CNN Feature Hierarchy for HSI Classification
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Papers With Code - HSI Classification](https://paperswithcode.com/task/hyperspectral-image-classification)
- [动手学深度学习](https://zh.d2l.ai/)