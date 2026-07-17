"""
HybridSN 模型定义 + 改进版本
参考论文: "HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy
           for Hyperspectral Image Classification"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础模块 ====================

class SELayer(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE3DLayer(nn.Module):
    """3D 版 Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock3D(nn.Module):
    """3D 残差块（含 SE 注意力可选）"""
    def __init__(self, in_ch, out_ch, kernel=(3, 3, 3), use_se=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.se = SE3DLayer(out_ch) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


# ==================== HybridSN 原始版本 ====================

class HybridSN(nn.Module):
    """
    HybridSN 标准版 (60分基线)

    输入: (B, C, H, W) 其中 C 为光谱波段数, H=W=window_size
    实际输入形状: (B, 1, C, H, W) — 会通过 unsqueeze 增加 channel 维度

    结构:
      - 3 层 3D 卷积: 8 → 16 → 32 通道
      - reshape 到 2D
      - 1 层 2D 卷积: 64 通道
      - Flatten → FC(256) → FC(num_classes)
    """
    def __init__(self, num_classes, input_channels=30, patch_size=25, dropout=0.4):
        """
        参数:
            num_classes:    类别数
            input_channels: PCA 降维后的波段数（默认 30）
            patch_size:     输入 patch 大小
            dropout:        Dropout 比例
        """
        super().__init__()

        # 3D 卷积部分
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3))
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3))
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3))
        self.bn3d_3 = nn.BatchNorm3d(32)

        # 计算 3D 卷积后的尺寸
        self._calc_3d_output_size(input_channels, patch_size)

        # 2D 卷积部分
        self.conv2d = nn.Conv2d(self.c2d_in, 64, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(64)

        # 全连接
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.fc_in, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def _calc_3d_output_size(self, C, S):
        """计算 3D 卷积输出尺寸"""
        c1 = C - 7 + 1    # conv1 (kernel=7, valid padding)
        c2 = c1 - 5 + 1   # conv2 (kernel=5, valid padding)
        c3 = c2 - 3 + 1   # conv3 (kernel=3, valid padding)

        s1 = S - 3 + 1
        s2 = s1 - 3 + 1
        s3 = s2 - 3 + 1

        self.c2d_in = 32 * c3   # 32 通道 × 剩余光谱维度
        self.fc_in = 64 * s3 * s3

    def forward(self, x):
        # x: (B, C, H, W) → (B, 1, C, H, W)
        x = x.unsqueeze(1)

        # 3D 卷积
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))

        # Reshape: (B, C, D, H, W) → (B, C*D, H, W)
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        # 2D 卷积
        x = F.relu(self.bn2d(self.conv2d(x)))

        # Flatten → FC
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.classifier(x)

        return x


# ==================== HybridSN-SE (改进版 A: 加注意力) ====================

class HybridSN_SE(nn.Module):
    """
    HybridSN + 通道注意力 (Squeeze-and-Excitation)
    在 2D 卷积后加入 SE 模块，增强重要通道响应
    """
    def __init__(self, num_classes, input_channels=30, patch_size=25, dropout=0.4):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3))
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.se3d_1 = SE3DLayer(8)

        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3))
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.se3d_2 = SE3DLayer(16)

        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3))
        self.bn3d_3 = nn.BatchNorm3d(32)
        self.se3d_3 = SE3DLayer(32)

        self._calc_3d_output_size(input_channels, patch_size)

        self.conv2d = nn.Conv2d(self.c2d_in, 64, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(64)
        self.se2d = SELayer(64)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.fc_in, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def _calc_3d_output_size(self, C, S):
        c1 = C - 7 + 1
        c2 = c1 - 5 + 1
        c3 = c2 - 3 + 1
        s1 = S - 3 + 1
        s2 = s1 - 3 + 1
        s3 = s2 - 3 + 1
        self.c2d_in = 32 * c3
        self.fc_in = 64 * s3 * s3

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = self.se3d_1(x)
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = self.se3d_2(x)
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))
        x = self.se3d_3(x)

        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        x = F.relu(self.bn2d(self.conv2d(x)))
        x = self.se2d(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.classifier(x)

        return x


# ==================== HybridSN-Res (改进版 B: 加残差+SE) ====================

class HybridSN_Res(nn.Module):
    """
    HybridSN + 残差连接 + SE 注意力
    将 3D 卷积替换为残差块，提升深层特征提取能力
    """
    def __init__(self, num_classes, input_channels=30, patch_size=25, dropout=0.4):
        super().__init__()

        self.res3d_1 = ResidualBlock3D(1, 8, kernel=(7, 3, 3), use_se=True)
        self.res3d_2 = ResidualBlock3D(8, 16, kernel=(5, 3, 3), use_se=True)
        self.res3d_3 = ResidualBlock3D(16, 32, kernel=(3, 3, 3), use_se=True)

        self._calc_3d_output_size(input_channels, patch_size)

        self.conv2d = nn.Conv2d(self.c2d_in, 64, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(64)
        self.se2d = SELayer(64)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.fc_in, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def _calc_3d_output_size(self, C, S):
        c1 = C - 7 + 1
        c2 = c1 - 5 + 1
        c3 = c2 - 3 + 1
        s1 = S - 3 + 1
        s2 = s1 - 3 + 1
        s3 = s2 - 3 + 1
        self.c2d_in = 32 * c3
        self.fc_in = 64 * s3 * s3

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.res3d_1(x)
        x = self.res3d_2(x)
        x = self.res3d_3(x)

        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        x = F.relu(self.bn2d(self.conv2d(x)))
        x = self.se2d(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.classifier(x)

        return x


# ==================== 快速测试 ====================

if __name__ == "__main__":
    print("=" * 55)
    print("  模型测试")
    print("=" * 55)

    for name, ModelCls in [
        ("HybridSN", HybridSN),
        ("HybridSN_SE", HybridSN_SE),
        ("HybridSN_Res", HybridSN_Res),
    ]:
        model = ModelCls(num_classes=16, input_channels=30, patch_size=25)
        x = torch.randn(2, 30, 25, 25)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n  {name}:")
        print(f"    输入: {list(x.shape)}")
        print(f"    输出: {list(y.shape)}")
        print(f"    参数量: {params:,}")

    print("\n  模型测试通过！✓")