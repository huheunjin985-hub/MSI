##########################################
#     【子空间投影层】
#     对应论文中的 T_RGB 矩阵。
#     目标：将多光谱数据 (High-Dim) 降维投影到 RGB 子空间 (Low-Dim)。
#     物理含义：模拟传感器光谱响应的线性组合，这步仅仅是降维，还没做白平衡。
#     注意：这里的参数是“可学习的”，训练完后通常是一个固定的矩阵。
##########################################

import torch
import torch.nn as nn
from config import SystemConfig


class SubspaceProjection(nn.Module):


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