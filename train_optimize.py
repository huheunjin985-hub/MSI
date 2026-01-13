##############################################
# 实现论文中的两阶段优化思路：光谱先验优化 + 神经网络训练
###############################################
import torch
import torch.optim as optim
from pipeline import MSIReproductionPipeline
from config import SystemConfig


class EnterpriseTrainer:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = MSIReproductionPipeline(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def calculate_losses(self, outputs, targets):
        """
        实现论文公式 (6), (7), (8)
        """
        # 1. Angular Error (AE) for Illuminance
        l_hat = outputs['l_hat']
        l_gt = targets['illuminance']

        # Cosine Similarity -> Angular Error
        cosine_sim = torch.nn.functional.cosine_similarity(l_hat, l_gt, dim=1)
        loss_ae = torch.mean(torch.acos(torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7)))

        # 2. Color Difference (Delta E 2000)
        # 这里为了简化代码使用 MSE/L1 代替，企业级实现需引入专门的 CIEDE2000 库
        img_pred = outputs['rgb_image']
        img_gt = targets['reference_rgb']
        loss_reproduction = torch.nn.functional.mse_loss(img_pred, img_gt)

        # 3. RMSE for Matrix
        # 如果有 M_GT (通过固定 T_RGB 优化得到的)，则计算此 Loss
        loss_rmse = 0.0
        if 'matrix_gt' in targets:
            loss_rmse = torch.sqrt(torch.nn.functional.mse_loss(outputs['m_final'], targets['matrix_gt']))

        total_loss = loss_reproduction + \
                     self.config.model.lambda_ae * loss_ae + \
                     self.config.model.lambda_rmse * loss_rmse

        return total_loss, {"ae": loss_ae.item(), "rep": loss_reproduction.item()}

    def train_step(self, batch_data):
        """
        单步训练接口
        """
        raw_msi = batch_data['msi'].to(self.config.device)

        self.optimizer.zero_grad()
        final_rgb, meta_data = self.model(raw_msi)

        outputs = {"rgb_image": final_rgb, **meta_data}
        loss, metrics = self.calculate_losses(outputs, batch_data)  # batch_data 需包含 GT

        loss.backward()
        self.optimizer.step()

        return metrics


# -------------------------------------------------------------
# 论文算法 1: Spectral Prior-guided Optimization (离线优化 T_RGB)
# -------------------------------------------------------------
def optimize_spectral_prior(config: SystemConfig):
    """
    根据论文 3.2 节：
    在神经网络训练之前，先通过凸优化或梯度下降
    求解最佳的 T_RGB 和初始 M
    """
    print("开始光谱先验优化...")
    # 1. 初始化 T_RGB (C x 3) 和 M (3 x 3)
    T = torch.nn.Parameter(torch.randn(config.camera.num_channels, 3))
    M = torch.nn.Parameter(torch.eye(3))
    opt = optim.Adam([T, M], lr=0.01)

    # 2. 模拟数据集 (X_i, L_i, Y_i)
    # 此处应加载 Synthetic Data (Algorithm 2 in paper)

    # 3. 循环优化
    # minimize Distance( (X @ T @ D(L@T) @ M), Y )
    # ... 实现略 ...

    print("优化完成，保存 T_RGB 和 M_static")
    return T.data, M.data