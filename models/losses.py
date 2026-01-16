import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperLoss(nn.Module):
    def __init__(self, lambda_ae=0.1, lambda_rmse=0.0):
        """
        严格对应论文公式 (6) 和 (7) 的损失函数
        Args:
            lambda_ae: 对应 train.py 中的 config.model.lambda_ae
            lambda_rmse: 对应 train.py 中的 config.model.lambda_rmse (尽管本阶段主用 AE)
        """
        super(PaperLoss, self).__init__()
        self.lambda_ae = lambda_ae
        self.lambda_rmse = lambda_rmse

        # 论文 Eq.6: Reproduction Error
        # 标准做法是使用 MSE (L2 Norm 的平方)，这符合物理模型拟合的常规
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_rgb, gt_rgb, pred_L, gt_L):
        """
        pred_rgb: (B, 3, H, W) - 预测图像
        gt_rgb: (B, 3, H, W) - 真值图像
        pred_L: (B, C) - 预测光照向量 l_hat
        gt_L: (B, C) - 真值光照向量
        """

        # --- 1. 图像重构损失 (Eq. 6) ---
        loss_rec = self.mse_loss(pred_rgb, gt_rgb)

        loss_ae_rad = torch.tensor(0.0, device=pred_rgb.device)
        ang_deg_display = 0.0

        # --- 2. 光照角度误差 (Eq. 7) ---
        if self.lambda_ae > 0:
            # 归一化 (Normalize)
            p_n = F.normalize(pred_L, p=2, dim=1)
            g_n = F.normalize(gt_L, p=2, dim=1)

            # 余弦相似度 (Dot Product)
            cosine = torch.sum(p_n * g_n, dim=1)

            # 数值稳定性截断 (Clamp): 防止 1.000001 导致 acos 出现 NaN
            cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

            # 计算弧度 Loss (Radians):
            # 数学上优化弧度最稳定 (0 ~ 3.14)，直接优化角度 (0 ~ 180) 会导致梯度爆炸
            loss_ae_rad = torch.mean(torch.acos(cosine))

            # 仅在不计算梯度时转换为角度，用于返回给日志打印
            with torch.no_grad():
                ang_deg_display = loss_ae_rad.item() * (180.0 / 3.1415926)

        # --- 总损失 ---
        # 按照论文逻辑组合: L_total = L_rec + λ * L_ang
        total_loss = loss_rec + (self.lambda_ae * loss_ae_rad)

        # 返回三个值：
        # 1. total_loss: 用于反向传播 (必须包含梯度)
        # 2. loss_rec.item(): 用于打印 Rec Loss 数值
        # 3. ang_deg_display: 用于打印角度误差 (度数)
        return total_loss, loss_rec.item(), ang_deg_display