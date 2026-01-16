import torch
import torch.nn as nn
from modules import IlluminanceEstimator, ColorCorrectionPredictor, SubspaceProjection
from config import SystemConfig


class MSIReproductionPipeline(nn.Module):

    def __init__(self, config: SystemConfig, initial_T=None, initial_M=None):
        """
        Args:
            config: 系统配置
            initial_T: (Tensor, optional) Stage 1 优化得到的 T_RGB 矩阵 (C x 3)
            initial_M: (Tensor, optional) Stage 1 优化得到的 M 矩阵 (3 x 3)
        """
        super().__init__()
        self.config = config

        # 1. T_RGB 模块 (SubspaceProjection)
        self.t_rgb_layer = SubspaceProjection(config)

        # [修改点 1] 如果提供了预计算的 T_RGB，直接加载进 Linear 层
        if initial_T is not None:
            print("⚡ 已加载 Stage 1 优化的 T_RGB 矩阵")
            # 注意维度：nn.Linear 的 weight 是 (Out, In)，也就是 (3, C)
            # 我们传进来的 initial_T 通常是 (C, 3)，所以需要转置 (.t())
            with torch.no_grad():
                self.t_rgb_layer.projection.weight.copy_(initial_T.t())

            # 可选：你可以选择是否冻结 T_RGB，通常建议设为 True 继续微调，或 False 锁死
            # self.t_rgb_layer.projection.weight.requires_grad = True

        # 2. 神经网络部分 (CNN + MLP)
        self.illuminance_net = IlluminanceEstimator(config)
        self.correction_mlp = ColorCorrectionPredictor(config)

        # 3. 静态 Color Correction Matrix (M_static)
        # [修改点 2] 如果提供了预计算的 M，直接初始化
        if initial_M is not None:
            print("⚡ 已加载 Stage 1 优化的 M_static 矩阵")
            self.m_static = nn.Parameter(initial_M, requires_grad=True)
        else:
            # 默认初始化为单位阵
            self.m_static = nn.Parameter(torch.eye(3), requires_grad=True)

    """
                    前向传播逻辑
                    Args:
                        raw_msi: 输入的多光谱 RAW 数据，形状 (Batch, C, H, W)
                    """
    def forward(self, raw_msi):
        B, C, H, W = raw_msi.shape # 1. 解包输入数据的维度  B: Batch Size, C: Channel数(如9), H: 高度, W: 宽度

        # 步骤 A: 照度估计 (Illuminance Estimation)
        # 目标: 获得推测的场景光源向量 L_hat
        l_hat, _ = self.illuminance_net(raw_msi)# 输入 (B, C, H, W) -> 输出 l_hat (B, C) 第二个返回值 (可能是特征图) 暂时不用，所以用 _ 接收

        # 步骤 B: RGB 子空间投影 (RGB Subspace Projection)
        # 目标: I_subspace = I_MSI * T_RGB
        msi_permuted = raw_msi.permute(0, 2, 3, 1)# PyTorch 的 nn.Linear 操作的是最后一个维度 所以先将通道维 C 移到最后: (B, C, H, W) -> (B, H, W, C)
        i_subspace = self.t_rgb_layer(msi_permuted)# 执行线性投影: (B, H, W, C) @ (C, 3) -> (B, H, W, 3) 这一步得到了“未白平衡”的粗略 RGB 图像

        # 步骤 C: 白平衡 (White Balancing - Von Kries Hypothesis)
        # 公式: D = diag(1 / (L * T_RGB))
        t_rgb_weights = self.t_rgb_layer.projection.weight.t()   # 1. 获取投影层的权重矩阵 T_RGB self.t_rgb_layer.projection.weight 形状是 (3, C)，需要转置回 (C, 3)
        w_coeffs = torch.matmul(l_hat, t_rgb_weights)   # 2. 计算白点 (White Point) W = L * T_RGB,  l_hat (B, C) @ t_rgb_weights (C, 3) -> w_coeffs (B, 3) ,这里的 w_coeffs 表示在当前 T_RGB 空间下，该光源对应的 RGB 值
        wb_gains = 1.0 / (w_coeffs + 1e-6)# 3. 计算增益 (Gain) = 1 / W 加上 1e-6 是为了防止分母为 0 (数值稳定性)
        wb_gains = wb_gains.view(B, 1, 1, 3)# 4. 调整形状以便广播 (Broadcasting) (B, 3) -> (B, 1, 1, 3)，这样可以自动应用到图像的每个像素 (H, W) 上
        i_wb = i_subspace * wb_gains# 5. 应用白平衡 Element-wise multiplication: (B,H,W,3) * (B,1,1,3)

        # 步骤 D: 色彩校正 (Color Correction)
        # 公式: M_final = M_static + M_residual(L)
        m_residual, _ = self.correction_mlp(l_hat)# 1. 使用 MLP 根据光照 L 预测动态残差矩阵 l_hat (B, C) -> m_residual (B, 3, 3)
        m_final = self.m_static.unsqueeze(0) + m_residual # 2. 组合最终矩阵 m_static 是 (3, 3)，先 unsqueeze 变成 (1, 3, 3) 以便和 batch 相加
        i_flat = i_wb.view(B, -1, 3)# 3. 执行矩阵乘法: Image * M_final ,由于 torch.matmul 只能处理最后两维的矩阵乘法，我们需要先把 H, W 拍平, (B, H, W, 3) -> (B, H*W, 3)
        rgb_flat = torch.matmul(i_flat, m_final)# (B, Pixel数, 3) @ (B, 3, 3) -> (B, Pixel数, 3), 这里的数学含义是：每个像素的 rgb 向量 (1x3) 乘以校正矩阵 (3x3)
        rgb_restored = rgb_flat.view(B, H, W, 3)# 4. 恢复图像原本的空间维度, (B, Pixel数, 3) -> (B, H, W, 3)

        # 步骤 E: 后处理与输出
        final_rgb = rgb_restored.permute(0, 3, 1, 2)# 1. 调整通道顺序，变回 PyTorch 标准格式, (B, H, W, 3) -> (B, 3, H, W)

        # 2. Gamma 校正 (仅用于将 Linear RGB 转为 sRGB 进行显示/计算Loss)
        # 使用 clamp 限制值在 0-1 之间，防止负值产生 NaN
        # 注意：这里使用了简化的 1/2.2 Gamma，如果您的数据生成用了标准 sRGB，
        # 在计算 Loss 时要确保 GT 也是同样的 Gamma 标准
        final_rgb = torch.clamp(final_rgb, 0, 1) ** (1 / 2.2)

        return final_rgb, {"l_hat": l_hat, "m_final": m_final} # 返回最终图像，以及用于计算中间 Loss 的变量

    def load_fixed_params(self, path_dict):
        """留空接口：用于加载固定参数"""
        pass