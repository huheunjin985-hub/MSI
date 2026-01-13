#########################################
# 核心组件：实现CNN估测器、MLP校正器和基础的数学变换。
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SystemConfig


class IlluminanceEstimator(nn.Module):
    """
    对应论文中的 CNN 部分：从 MSI 估计照度向量 L (C x 1)
    【光照估计器】
    对应论文中的 CNN 部分 (CSPDNet 的简化版或等效实现)。
    目标：从输入的多光谱图像 (MSI) 中提取场景的全局光照特征。
    物理含义：估计场景光源的光谱功率分布 L (C x 1)。
    """

    def __init__(self, config: SystemConfig):
        super().__init__()
        # 获取输入图像的通道数 (例如 ms_channels = 9)
        c_in = config.camera.num_channels

        # 定义特征提取网络 (Sequential 容器)
        # 简单复现论文思路，实际项目中可以替换为更复杂的骨干网络 (如 ResNet, CSPDarknet 等
        self.features = nn.Sequential(
            # 第一层卷积: 提取低级光谱-空间特征
            # 输入: (Batch, c_in, H, W) -> 输出: (Batch, 32, H, W)
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1),

            # 激活函数: 引入非线性 (inplace=True 节省显存)
            nn.ReLU(inplace=True),

            # 下采样: 减小特征图尺寸，扩大感受野
            # 输出: (Batch, 32, H/2, W/2)
            nn.MaxPool2d(2),

            # 第二层卷积: 进一步提取高级特征
            # 输入: (Batch, 32, H/2, W/2) -> 输出: (Batch, 64, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            # 激活函数
            nn.ReLU(inplace=True),

            # 全局平均池化 (Global Average Pooling)
            # 作用：消除空间维度信息，因为光照通常被认为是全局统一的 (Global Illumination Assumption)
            # 输入: (Batch, 64, H/2, W/2) -> 输出: (Batch, 64, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1))  # Global Pooling
        )
        # 回归层 (全连接层)
        # 作用：将提取的 64 维特征映射回光谱通道维度，得到光照向量
        # 输入: 64 -> 输出: c_in (例如 9)
        self.regressor = nn.Linear(64, c_in)  # 输出 C 个通道的照度值

    def forward(self, x):
        """
        前向传播函数
        Input: (B, C, H, W) - 批次大小, 通道数, 高, 宽
        Output: (B, C) - 估计出的归一化光照向量 L
        """
        # 1. 提取特征
        # feat shape: (B, 64, 1, 1)
        feat = self.features(x)

        # 2. 展平 (Flatten)
        # 将 (B, 64, 1, 1) 展平为 (B, 64)，以便送入全连接层
        feat = feat.view(feat.size(0), -1)

        # 3. 回归预测
        # l_hat shape: (B, C)
        l_hat = self.regressor(feat)

        # 4. 归一化 (L2 Normalize)
        # 物理意义：我们在乎光照的“颜色/光谱成分”而不是“强度”。
        # 比如：我们想知道光源是“偏红”还是“偏蓝”，而不关心它有多亮。
        # 公式: v = v / ||v||_2
        l_hat = F.normalize(l_hat, p=2, dim=1)

        # 返回估计的光照向量
        return l_hat


class ColorCorrectionPredictor(nn.Module):
    """
    【色彩校正预测器】
    对应论文中的 MLP 部分：从 L 预测动态残差矩阵 (3x3)
    目标：根据估计出的光照向量，动态生成色彩校正矩阵 (CCM)。
    物理含义：不同光照下，我们需要微调 RGB 的混合比例，这个 MLP 负责算出这个“微调量” (ΔM)。
    """

    def __init__(self, config: SystemConfig):
        # 初始化父类
        super().__init__()

        # 输入维度：光照向量的长度 (即相机通道数 C)
        c_in = config.camera.num_channels

        # 隐藏层维度：通常在 config 中定义 (例如 64 或 128)
        h_dim = config.model.hidden_dim

        # 定义多层感知机 (MLP)
        self.mlp = nn.Sequential(

            # 第一层全连接：将光照向量映射到隐藏空间
            nn.Linear(c_in, h_dim),

            # 激活函数
            nn.ReLU(),

            # 第二层全连接：映射到 9 个输出值
            # 为什么是 9？因为我们要生成一个 3x3 的矩阵，总共 9 个元素。
            #组照度无关的矩阵M
            nn.Linear(h_dim, 9)  # 输出 3x3 = 9 个元素
        )

    def forward(self, l_vector):
        """
        Input: (B, C) -> l_vector
        Output: (B, 3, 3) -> Residual Matrix
        Input: (B, C) -> l_vector (上一步 IlluminanceEstimator 的输出)
        Output: (B, 3, 3) -> Residual Matrix (残差矩阵 ΔM)
        """
        # 1. 通过 MLP 计算
        # delta_m shape: (B, 9)
        delta_m = self.mlp(l_vector)

        # 2. 重塑形状 (Reshape)
        # 将平铺的 9 个值变回 3x3 的矩阵形式
        # shape: (B, 9) -> (B, 3, 3)
        delta_m = delta_m.view(-1, 3, 3)

        # 返回残差矩阵
        return delta_m


class SubspaceProjection(nn.Module):
    """
    【子空间投影层】
    对应论文中的 T_RGB 矩阵。
    目标：将多光谱数据 (High-Dim) 降维投影到 RGB 子空间 (Low-Dim)。
    物理含义：模拟传感器光谱响应的线性组合，这步仅仅是降维，还没做白平衡。
    注意：这里的参数是“可学习的”，训练完后通常是一个固定的矩阵。
    """

    def __init__(self, config: SystemConfig):
        super().__init__()
        # T_RGB shape: (C, 3)
        # 使用 Linear 实现矩阵乘法: x @ T
        # 定义一个线性层 (Linear Layer)
        # 该层本质上就是一个矩阵乘法：Input @ Weight.T
        # 输入维度: multispectral channels (C)
        # 输出维度: RGB channels (3)
        # bias=False: 因为是色彩空间变换，通常只要旋转/缩放，不需要偏置偏移 (原点应保持为黑色)
        # T_RGB 矩阵的形状实际上存储在 self.projection.weight 中，形状为 (3, C)
        self.projection = nn.Linear(config.camera.num_channels, 3, bias=False)

    def forward(self, x):
        """
                Input: (B, H, W, C) - 注意这里通常假定通道在最后，或者需要外面调整
                Output: (B, H, W, 3) - 投影后的 RGB 粗略图像
                """
        # x: (B, H, W, C)
        # 执行线性投影
        # PyTorch 的 Linear 层会自动对输入的最后一个维度进行计算
        # 计算公式: x_rgb = x_msi * T_RGB
        return self.projection(x)