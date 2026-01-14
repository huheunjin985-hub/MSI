"""
光照估计器：基于CNN的场景照度向量预测
支持两种工作模式：
1. 预测模式 (默认): 使用CNN估计照度向量
2. 真实数据模式: 使用外部传入的真实照度向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SystemConfig

class IlluminanceEstimator(nn.Module):


    def __init__(self, config: SystemConfig):
        super().__init__()
        c_in = config.camera.num_channels

        # 基础特征提取网络
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )

        # 回归层：从特征映射到照度向量
        self.regressor = nn.Linear(64, c_in)

        # 配置参数
        self.config = config
        self.current_mode = 'predict'  # 'predict' or 'gt'
        self.register_buffer('ground_truth_illuminance', None)

    def forward(self, x, ground_truth=None):
        """
        前向传播函数
        Args:
            x: (B, C, H, W) - 输入多光谱图像
            ground_truth: (B, C) - 可选，真实照度向量
        Returns:
            l_hat: (B, C) - 估计的归一化照度向量
            features: (B, 64) - 提取的中间特征（用于调试和可视化）
        """
        # 根据模式处理输入
        if ground_truth is not None:
            self.current_mode = 'gt'
            l_hat = ground_truth
        elif self.current_mode == 'gt' and self.ground_truth_illuminance is not None:
            l_hat = self.ground_truth_illuminance
        else:
            self.current_mode = 'predict'
            # 提取特征
            feat = self.features(x)
            feat = feat.view(feat.size(0), -1)
            # 回归预测
            l_hat = self.regressor(feat)

        # 归一化（无论是否使用真实数据都执行此操作）
        l_hat = F.normalize(l_hat, p=2, dim=1)

        # 返回中间特征供后续分析
        if self.current_mode == 'predict':
            return l_hat, feat
        else:
            # 如果使用真实数据，返回零向量作为特征占位符
            return l_hat, torch.zeros(x.size(0), 64, device=x.device)

    def set_ground_truth(self, illuminance):
        """设置全局真实照度向量（用于测试）"""
        self.ground_truth_illuminance = illuminance
        self.current_mode = 'gt'

    def reset_to_predict_mode(self):
        """重置为预测模式"""
        self.current_mode = 'predict'

    @staticmethod
    def calculate_angular_error(pred, target):
        """计算论文中提到的角误差 AE"""
        pred_normalized = F.normalize(pred, dim=1)
        target_normalized = F.normalize(target, dim=1)
        cosine_sim = torch.sum(pred_normalized * target_normalized, dim=1)
        # 限制范围以避免数值不稳定
        cosine_sim = torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7)
        return torch.acos(cosine_sim)

    @classmethod
    def from_pretrained(cls, config: SystemConfig, checkpoint_path):
        """加载预训练模型"""
        model = cls(config)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['illuminance_estimator'])
        return model