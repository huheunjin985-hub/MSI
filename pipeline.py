#####################################################
# 完整流水线：严格按照公式组装
########################################################
import torch
import torch.nn as nn
from components import IlluminanceEstimator, ColorCorrectionPredictor, SubspaceProjection
from config import SystemConfig


class MSIReproductionPipeline(nn.Module):
    """
    完整的色彩再现系统
    """

    def __init__(self, config: SystemConfig):
        super().__init__()
        self.config = config

        # 1. T_RGB 模块
        self.t_rgb_layer = SubspaceProjection(config)

        # 2. 神经网络部分 (CNN + MLP)
        self.illuminance_net = IlluminanceEstimator(config)
        self.correction_mlp = ColorCorrectionPredictor(config)

        # 3. 静态 Color Correction Matrix (M_static)
        # 初始化为单位阵，训练中会更新或加载预训练
        self.m_static = nn.Parameter(torch.eye(3), requires_grad=True)

    def forward(self, raw_msi):
        """
        执行完整的颜色校正流程
        Args:
            raw_msi: (B, C, H, W) 输入的 RAW 多光谱图像
        Returns:
            final_rgb: (B, 3, H, W) 校正后的 sRGB 图像
            data_dict: 包含中间变量用于 Loss 计算 (L_hat, M_final)
        """
        B, C, H, W = raw_msi.shape

        # --- 步骤 A: 照度估计 (Illuminance Estimation) ---
        # l_hat: (B, C)
        l_hat = self.illuminance_net(raw_msi)

        # --- 步骤 B: RGB 子空间投影 (RGB Subspace Projection) ---
        # 调整维度以适应 Linear 层: (B, H, W, C)
        msi_permuted = raw_msi.permute(0, 2, 3, 1)
        # i_subspace: (B, H, W, 3) -> 对应公式 I_MSI * T_RGB
        i_subspace = self.t_rgb_layer(msi_permuted)

        # --- 步骤 C: 计算白平衡增益 (White Balancing) ---
        # 获取 T_RGB 权重: (3, C) -> 转置为 (C, 3)
        t_rgb_weights = self.t_rgb_layer.projection.weight.t()

        # 计算白平衡系数 W = L_hat * T_RGB
        # (B, C) @ (C, 3) -> (B, 3)
        w_coeffs = torch.matmul(l_hat, t_rgb_weights)

        # 为了应用白平衡，我们需要除以光源颜色 (Von Kries 假设)
        # D = diag(1/W) 或者直接应用增益。
        # 注意：论文中 D=diag(W) 可能是指应用 W 去归一化，
        # 但通常 WB 是 I_balanced = I / W。这里我们生成增益 Gain = 1 / (W + eps)
        wb_gains = 1.0 / (w_coeffs + 1e-6)
        wb_gains = wb_gains.view(B, 1, 1, 3)  # 广播到 (B, H, W, 3)

        # 应用白平衡: I_wb = I_subspace * D
        i_wb = i_subspace * wb_gains

        # --- 步骤 D: 动态色彩校正 (Dynamic Color Correction) ---
        # 预测残差矩阵: (B, 3, 3)
        m_residual = self.correction_mlp(l_hat)

        # 最终矩阵 M_L = M_static + M_residual
        # 广播 M_static: (1, 3, 3)
        m_final = self.m_static.unsqueeze(0) + m_residual

        # 应用矩阵校正: I_target = I_wb * M_final
        # (B, H, W, 3) @ (B, 3, 3) -> (B, H, W, 3)
        # 注意矩阵乘法顺序，通常 Image 视为行向量集合 I x M
        final_rgb = torch.matmul(i_wb, m_final)

        # 转回 (B, 3, H, W) 用于输出或后续处理
        final_rgb = final_rgb.permute(0, 3, 1, 2)

        # --- 步骤 E: Gamma 校正 (Visualization) ---
        # 简单 sRGB Gamma，实际部署可能需要更复杂的 Tone Mapping
        final_rgb = torch.clamp(final_rgb, 0, 1) ** (1 / 2.2)

        return final_rgb, {"l_hat": l_hat, "m_final": m_final}

    def load_fixed_params(self, path_dict):
        """接口：加载论文中提到的固定参数"""
        pass