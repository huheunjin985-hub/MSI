import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SystemConfig


class ColorCorrectionPredictor(nn.Module):
    """
    色彩校正预测器：基于MLP的动态残差矩阵预测
    支持两种工作模式：
    1. 预测模式 (默认): 使用MLP预测残差矩阵
    2. 真实数据模式: 使用外部传入的真实残差矩阵
    """

    def __init__(self, config: SystemConfig):
        super().__init__()
        c_in = config.camera.num_channels
        h_dim = config.model.hidden_dim

        # 多层感知机网络
        self.mlp = nn.Sequential(
            nn.Linear(c_in, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 9)  # 输出3x3矩阵
        )

        # 配置参数
        self.config = config
        self.current_mode = 'predict'  # 'predict' or 'gt'
        self.register_buffer('ground_truth_matrix', None)

    def forward(self, l_vector, ground_truth=None):
        """
        前向传播函数
        Args:
            l_vector: (B, C) - 输入光照向量
            ground_truth: (B, 3, 3) - 可选，真实残差矩阵
        Returns:
            delta_m: (B, 3, 3) - 预测或传入的残差矩阵
            hidden_repr: (B, h_dim) - 隐藏层表示（用于调试）
        """
        # 根据模式处理输入
        if ground_truth is not None:
            self.current_mode = 'gt'
            delta_m = ground_truth
        elif self.current_mode == 'gt' and self.ground_truth_matrix is not None:
            delta_m = self.ground_truth_matrix
        else:
            self.current_mode = 'predict'
            # 通过MLP预测残差矩阵
            delta_m = self.mlp(l_vector)
            delta_m = delta_m.view(-1, 3, 3)

        # 返回隐藏层表示供后续分析
        if self.current_mode == 'predict':
            # 获取隐藏层输出
            hidden_repr = self._get_hidden_representation(l_vector)
            return delta_m, hidden_repr
        else:
            # 如果使用真实数据，返回零向量作为隐藏层表示
            return delta_m, torch.zeros(l_vector.size(0), self.config.model.hidden_dim, device=l_vector.device)

    def _get_hidden_representation(self, l_vector):
        """获取MLP的隐藏层输出"""
        x = l_vector
        for i, layer in enumerate(self.mlp.children()):
            x = layer(x)
            if i == 1:  # 第一个ReLU之后
                break
        return x

    def set_ground_truth(self, matrix):
        """设置全局真实残差矩阵（用于测试）"""
        self.ground_truth_matrix = matrix
        self.current_mode = 'gt'

    def reset_to_predict_mode(self):
        """重置为预测模式"""
        self.current_mode = 'predict'

    @staticmethod
    def calculate_matrix_rmse(pred, target):
        """计算论文中提到的矩阵RMSE损失"""
        return torch.sqrt(F.mse_loss(pred.view(pred.size(0), -1), target.view(target.size(0), -1)))

    @classmethod
    def from_pretrained(cls, config: SystemConfig, checkpoint_path):
        """加载预训练模型"""
        model = cls(config)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['color_correction_predictor'])
        return model