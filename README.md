# 🌈 高光谱图像智能解译 — 基于 HybridSN 的分类大作业

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 目录

1. [什么是高光谱图像？](#-什么是高光谱图像)
2. [这个项目做什么？](#-这个项目做什么)
3. [整体流程（一张图看懂）](#-整体流程一张图看懂)
4. [环境配置](#-环境配置)
5. [数据集说明](#-数据集说明)
6. [代码文件详解](#-代码文件详解)
7. [怎么运行？](#-怎么运行)
8. [模型保存与复用](#-模型保存与复用)
9. [实验结果](#-实验结果)
10. [评估指标是什么意思？](#-评估指标是什么意思)
11. [项目结构](#-项目结构)
12. [常见问题](#-常见问题)

---

## 🤔 什么是高光谱图像？

### 通俗理解

普通照片只有**红、绿、蓝 3 个颜色通道**。你把相机对准一片森林，拍出来是绿色的。

高光谱图像有**几百个通道**，每一个通道对应一段极窄的光波波长。同样的森林，你用高光谱相机拍，能区分出"这片绿色是松树，那片绿色是杂草，旁边那块看起来一样绿但其实是灌木"。

这就是为什么高光谱被广泛应用在**遥感、农业、地质勘探、军事侦查**上——它能看到人眼看不到的物质差异。

---

## 🎯 这个项目做什么？

**一句话**：输入高光谱图像的一个小方块（25×25 像素），用深度学习模型判断这个方块中心像素属于哪一种地物类别。

项目完成以下工作：

1. **读数据** — 加载 `.mat` 格式的高光谱数据文件和对应的标签文件
2. **降维** — 原始数十至数百个波段太多、计算太慢，用 PCA 压缩到核心波段
3. **切块** — 以每个有标签的像素为中心，切出 25×25 的小方块作为训练样本
4. **建模型** — 搭建 HybridSN 神经网络（3D 卷积 + 2D 卷积混合架构，含 SE 注意力与残差改进版）
5. **训练** — 70% 训练 / 10% 验证 / 20% 测试，加权损失函数处理类别不均衡
6. **评估** — 输出 OA、AA、Kappa 系数、混淆矩阵及 7 张可视化图表

---

## 🔄 整体流程（一张图看懂）

```
高光谱数据 (H×W×C)           标签文件 (H×W)
        │                          │
        └──────────┬───────────────┘
                   ▼
             PCA 降维
                   │
                   ▼
         提取 Patches (N个, 每个 25×25×k)
                   │
                   ▼
      ┌────── 数据划分 (分层采样) ──────┐
      │         │              │        │
      ▼         ▼              ▼        │
   训练集     验证集         测试集      │
   (70%)     (10%)          (20%)      │
      │         │              │        │
      └─────────┴──────────────┘        │
                │                       │
                ▼                       │
      HybridSN 神经网络                 │
      ┌──────────────────────┐          │
      │ 3D Conv (光谱+空间)  │          │
      │ 3D Conv (光谱+空间)  │          │
      │ 3D Conv (光谱+空间)  │          │
      │ reshape → 2D Conv    │          │
      │ Flatten → FC → 分类  │          │
      └──────────────────────┘          │
                │                       │
                ▼                       │
          训练 → 验证 → 保存最佳模型    │
      (加权交叉熵 + Adam + LR调度)      │
                │                       │
                ▼                       │
      ┌──── 最终评估 ────┐              │
      │ OA / AA / Kappa  │◄─────────────┘
      │ 混淆矩阵 + 7张图  │
      └─────────────────┘
```

---

## 💻 环境配置

### 需要安装的 Python 库

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn h5py
```

| 库名 | 作用 |
|------|------|
| `torch` | **PyTorch 深度学习框架**，搭建和训练神经网络的核心 |
| `numpy` | **矩阵和数组运算**，数据存储和变换的基石 |
| `matplotlib` | **画图**（训练曲线、分类结果图） |
| `scipy` | **读取 .mat 文件**（v5 格式） |
| `h5py` | **读取 .mat 文件**（v7.3 格式，如 Houston 数据集） |
| `scikit-learn` | **机器学习工具**：PCA 降维、数据划分、评估指标 |

### 验证环境

```bash
python check_env.py
```

---

## 📦 数据集说明

### 支持的三个经典高光谱数据集

| 数据集 | 图像尺寸 | 波段数 | 类别数 | 标注像素数 | 场景 |
|--------|---------|--------|--------|-----------|------|
| **Indian Pines** | 145×145 | 200 | 16 | 10,249 | 农田/森林 |
| **Pavia University** | 610×340 | 103 | 9 | 42,776 | 城市遥感 |
| **Houston** | 210×954 | 48 | 7 | 2,530 | 城市遥感 |

### 文件格式

高光谱数据以 **MATLAB `.mat` 格式**存储，每个数据集需要两个文件：

- **数据文件**：三维数组 `(H, W, C)`，H 行 × W 列像素，C 个光谱波段
- **标签文件**：二维数组 `(H, W)`，每个像素的值即类别编号（0 表示未标注背景）

### 数据目录结构

```
data/
├── Indian Pines/
│   ├── Indian_pines_corrected.mat    ← 数据
│   └── Indian_pines_gt.mat           ← 标签
├── PaviaU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
└── Houston/
    ├── Houston13.mat                 ← v7.3 格式，需 h5py
    └── Houston13_7gt.mat
```

代码会自动在 `data/{数据集名}/` 下查找对应文件。

---

## 📝 代码文件详解

### `data_loader.py` — 数据加载与预处理

<details>
<summary><b>点击展开：6 个核心函数详解</b></summary>

#### `load_dataset(name, data_dir)` — 加载数据集

```python
data, gt, num_classes = load_dataset("IndianPines", "./data")
```

按数据集名称到 `data/{Indian Pines,Houston,PaviaU}/` 下查找 `.mat` 文件并读取。自动处理两种 MATLAB 格式（v5 用 scipy、v7.3 用 h5py）。

#### `pad_with_zeros(data, margin)` — 边界零填充

边缘像素切 patch 时会越界，先在外围填一圈 0 解决。

#### `create_patches(data, gt, window_size=25)` — 提取 Patches

遍历每个有标签的像素，切出 25×25 方块。跳过背景（标签=0），把原始标签重新映射为 0~N-1。同时返回每个样本在原图中的 `(row, col)` 坐标。

#### `apply_pca(data, n_components=30)` — PCA 降维

把 C 个光谱波段压缩到 k 个主成分。200→30 保留 99.8% 信息，48→20 保留 99.99%。

#### `HyperSpectralDataset` — PyTorch Dataset 包装

把 numpy `(H,W,C)` 转为 `(C,H,W)` 供 PyTorch 卷积使用。

#### `create_data_loaders(...)` — 划分 + 标准化 + DataLoader

分层采样保证每类在各子集中比例一致 → StandardScaler 标准化（只统计训练集）→ 打包 batch。

</details>

---

### `model.py` — 神经网络模型定义

<details>
<summary><b>点击展开：基础组件 + 3 个模型结构</b></summary>

#### 基础组件

- **SELayer**：Squeeze-and-Excitation 通道注意力。对 2D 特征图做全局平均池化→FC→Sigmoid→乘回原图，自动学习各通道重要性权重。
- **SE3DLayer**：3D 版本，在 (C, D, H, W) 上操作。
- **ResidualBlock3D**：3D 残差块 + SE 注意力。`out = conv(conv(x)) + shortcut(x)`，维度不变，解决深层网络梯度退化。

#### 模型

| 模型 | 说明 | 参数量 | 分数段 |
|------|------|--------|--------|
| `HybridSN` | 标准版（3 层 3D Conv → 1 层 2D Conv → FC） | 6.3M | **60** |
| `HybridSN_SE` | 标准版 + 各层后加 SE 注意力 | 6.3M | **80** |
| `HybridSN_Res` | 3D Conv 替换为残差块 + SE | 10.9M | **90+** |

</details>

---

### `train.py` — 训练与评估脚本

<details>
<summary><b>点击展开：5 步训练流程 + 关键设计</b></summary>

```
步骤1: 加载数据（load_dataset → create_patches → PCA → create_data_loaders）
步骤2: 创建模型（HybridSN / SE / Res，根据 --model 参数）
步骤3: 训练循环（train_one_epoch → evaluate → ReduceLROnPlateau 调度）
步骤4: 测试（加载最佳 checkpoint → 全量评估）
步骤5: 保存（*.pth 模型权重 + 可视化图表）
```

**关键设计决策**：

- **加权交叉熵损失**：`weight = 1/样本数`，小类犯错罚更重，防止模型偏向大类。
- **Adam 优化器**：自适应学习率，动量 + RMSProp。
- **ReduceLROnPlateau**：验证 loss 连续 10 epoch 不降 → lr 自动减半，精细搜索最优解。
- **最佳模型保存**：只保存验证准确率最高的 checkpoint，防止过拟合。

</details>

#### 可调参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | IndianPines | 数据集 (IndianPines / PaviaU / Houston) |
| `--model` | hybridsn | 模型 (hybridsn / hybridsn_se / hybridsn_res) |
| `--pca` | 30 | PCA 降维后波段数 |
| `--window_size` | 25 | 输入 patch 边长 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 0.001 | 初始学习率 |
| `--dropout` | 0.4 | Dropout 比例 |
| `--weight_decay` | 0.0001 | L2 正则化 |
| `--train_ratio` | 0.7 | 训练集占比 |
| `--seed` | 42 | 随机种子 |

---

### `visualize.py` — 可视化模块

训练过程中自动生成 **7 张高质量图表**，保存在 `results/figures/`。

| # | 文件名 | 内容 |
|---|--------|------|
| 1 | `01_data_overview.png` | 三栏：伪彩色合成 + PCA 第一主成分 + 标签图（含图例） |
| 2 | `02_spectral_curves.png` | 各类地物的平均光谱反射率曲线 |
| 3 | `03_class_distribution.png` | 各类标注像素数量柱状图 |
| 4 | `04_pca_variance.png` | PCA 累计解释方差曲线，标注降维维度和 99% 线 |
| 5 | `05_training_curves.png` | 训练/验证 Loss + Accuracy 双曲线，标注最佳 epoch |
| 6 | `06_confusion_matrix.png` | 归一化混淆矩阵 + 各类准确率柱状图 |
| 7 | `07_classification_map.png` | 真值 vs 预测 vs 对/错对照 + 图例 |

也可独立运行测试：

```bash
python visualize.py
```

---

### `test_model.py` — 模型测试与可视化

**加载已训练的 .pth 模型，无需重新训练**，直接进行测试和可视化。

```bash
# 基本用法
python test_model.py --model_path results/hybridsn_IndianPines.pth --num_samples 10

# 跨数据集
python test_model.py --model_path results/hybridsn_Houston.pth --dataset Houston --pca 20
```

| 步骤 | 说明 |
|------|------|
| 1️⃣ 加载模型 | 读取 .pth 文件，输出配置和保存时的准确率 / Kappa |
| 2️⃣ 加载数据 | PCA 降维 + StandardScaler 标准化 |
| 3️⃣ 全量评估 | 预测全部标注像素，输出 OA / AA / Kappa |
| 4️⃣ 随机抽样 | 抽取 N 个样本，打印位置、真实标签、预测标签、置信度 |
| 5️⃣ 可视化 | 随机样本图 + 混淆矩阵 + 分类结果图 |

---

## 🚀 怎么运行？

### 训练

```bash
# 60 分基线
python train.py --model hybridsn --epochs 10

# 80 分改进（SE 注意力）
python train.py --model hybridsn_se --epochs 50

# 90+ 分改进（残差 + SE）
python train.py --model hybridsn_res --epochs 50

# 不同数据集
python train.py --dataset Houston --pca 20 --epochs 100
python train.py --dataset PaviaU --pca 30 --epochs 100
```

### 测试（加载已训练模型，不重训）

```bash
python test_model.py --model_path results/hybridsn_IndianPines.pth --num_samples 10
```

### 模块独立测试

```bash
python data_loader.py    # 数据加载测试
python model.py          # 模型前向传播测试
python visualize.py      # 可视化模块测试
python check_env.py      # 依赖库检查
```

---

## 💾 模型保存与复用

训练完后 `results/` 下会生成 `.pth` 文件，包含模型权重、配置、标签映射和评估指标。后续无需重新训练：

```bash
python test_model.py --model_path results/hybridsn_IndianPines.pth
```

脚本自动从文件名识别模型类型（HybridSN / SE / Res），加载权重后直接推理。

---

## 📊 实验结果

### 跨数据集对比

| 数据集 | 图像尺寸 | 波段数 | 类别数 | OA | AA | Kappa | Epoch |
|--------|---------|--------|--------|-----|------|-------|-------|
| **Indian Pines** | 145×145 | 200→30 | 16 | **99.32%** | **99.56%** | **0.9922** | 10 |
| **Pavia University** | 610×340 | 103→30 | 9 | **99.96%** | **99.97%** | **0.9995** | 5 |
| **Houston** | 210×954 | 48→20 | 7 | **99.80%** | **99.80%** | **0.9977** | 100 |

> 三个数据集均使用 **HybridSN 标准版** + Adam + 加权交叉熵，CPU 训练。

### 分类结果可视化

#### Indian Pines (10 epoch, OA=99.32%)

![Indian Pines 训练曲线](results/figures/05_training_curves.png)

![Indian Pines 光谱曲线](results/figures/02_spectral_curves.png)

#### Houston (100 epoch, OA=99.80%)

![Houston 混淆矩阵](results/figures/06_confusion_matrix.png)

![Houston 分类结果](results/figures/07_classification_map.png)

### Indian Pines — 每类详细准确率

| 类别 | 原始标签 | 测试样本 | 准确率 |
|------|----------|---------|--------|
| 0 | 苜蓿 (Alfalfa) | 9 | 100.00% |
| 1 | 玉米-免耕 | 286 | 99.65% |
| 2 | 玉米-少耕 | 166 | 100.00% |
| 3 | 玉米 | 47 | 100.00% |
| 4 | 草地-牧场 | 97 | 100.00% |
| 5 | 草地-树木 | 146 | 99.32% |
| 6 | 草地-修剪 | 5 | 100.00% |
| 7 | 干草堆 | 96 | 100.00% |
| 8 | 燕麦 | 4 | 100.00% |
| 9 | 大豆-免耕 | 194 | 98.97% |
| 10 | 大豆-少耕 | 491 | 98.78% |
| 11 | 大豆-清理 | 119 | 97.48% |
| 12 | 小麦 | 41 | 100.00% |
| 13 | 森林 | 253 | 100.00% |
| 14 | 建筑-草地-树木 | 77 | 98.70% |
| 15 | 石头-铁塔 | 19 | 100.00% |

---

## 📐 评估指标是什么意思？

| 指标 | 通俗解释 |
|------|----------|
| **OA** (Overall Accuracy) | 预测正确的样本数 ÷ 总样本数。"所有题一共对了多少" |
| **AA** (Average Accuracy) | 各类准确率的平均值。对少数类更公平 |
| **Kappa** | 衡量分类结果比瞎猜好多少。Kappa=1 完美，0=瞎猜，<0=比瞎猜还差 |
| **混淆矩阵** | N×N 表格，第 i 行第 j 列 = 实际为 i 类但被预测为 j 类的数量。对角线越亮越好 |

---

## 📁 项目结构

```
能工智人大作业/
│
├── data_loader.py          ← 数据加载、PCA 降维、DataLoader
├── model.py                ← HybridSN + SE + 残差改进
├── train.py                ← 训练 → 验证 → 测试 → 保存模型
├── test_model.py           ← 加载 .pth，随机抽样测试 + 可视化
├── visualize.py            ← 7 张图表自动生成
├── check_env.py            ← 依赖库检查
│
├── data/                   ← 三个数据集
│   ├── Indian Pines/
│   ├── PaviaU/
│   └── Houston/
│
├── results/                ← 训练/测试输出
│   ├── figures/            ← 可视化图表 (7 PNG)
│   ├── *.pth               ← 模型权重（可复用）
│   └── *_report.txt        ← 文本分类报告
│
├── .gitignore
└── README.md
```

---

## ❓ 常见问题

### Q: 训练太慢怎么办？
- CPU：Indian Pines 约 50s/epoch，Houston 约 6s/epoch
- 先用 `--epochs 10` 快速验证，挂机跑完整训练
- NVIDIA 显卡可装 CUDA 版 PyTorch，快 10~30 倍

### Q: 训练完的模型怎么复用？
```bash
python test_model.py --model_path results/hybridsn_IndianPines.pth --num_samples 10
```
不需要重新训练，直接加载 `.pth` 测试。

### Q: 怎么知道我冲哪个分数段？

| 模型 | 分数 |
|------|------|
| `hybridsn` — 跑通基线 | **60** |
| `hybridsn_se` — 加注意力 | **80** |
| `hybridsn_res` — 加残差+注意力 | **90+** |