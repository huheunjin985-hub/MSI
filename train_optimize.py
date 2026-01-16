# train_optimize.py
import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from datasets.data_synthesis import SyntheticSpectralDataset


# ------------------------------------------------------------------
# 🔥 修改说明:
# 1. 删除了 EnterpriseTrainer 类 (因为训练逻辑现在在 train.py 里)
# 2. 增加了“缓存”机制：运行过一次后会自动保存，下次直接加载，不用重跑
# ------------------------------------------------------------------

def run_stage_1_optimization(config, hsi_data_path):
    """
    实现论文 Algorithm 1: Spectral Prior-guided Optimization (Stage 1)

    该函数会被 train.py 调用。
    如果检测到已经优化过的参数文件，直接加载返回；否则开始优化。
    """

    # 定义保存路径，避免重复计算
    save_dir = "checkpoints/priors"
    save_path = os.path.join(save_dir, "spectral_prior.pth")

    # --- 1. 检查是否存在缓存 ---
    if os.path.exists(save_path):
        print(f"⚡ [Stage 1] 发现已优化的物理先验: {save_path}")
        print("   👉 直接加载，跳过优化步骤...")
        data = torch.load(save_path, map_location=config.device)
        return data['t_rgb'], data['m_static']

    # --- 2. 如果没有缓存，开始优化 ---
    print(f"🚀 [Stage 1] 未找到缓存，开始运行光谱先验优化 (Optimization for T_RGB & M)...")

    device = config.device

    # 数据集准备 (只需要少量数据即可收敛)
    dataset = SyntheticSpectralDataset(hsi_data_root=hsi_data_path, config=config)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 参数初始化
    C = config.camera.num_channels
    # T_RGB: (C, 3)
    T_RGB = torch.nn.Parameter(torch.randn(C, 3, device=device) * 0.1, requires_grad=True)
    # M: (3, 3) 初始化为单位矩阵
    M = torch.nn.Parameter(torch.eye(3, device=device), requires_grad=True)

    optimizer = optim.Adam([T_RGB, M], lr=0.01)

    # 只需要跑 500 次迭代即可
    num_steps = 500
    iter_loader = iter(dataloader)

    for step in range(num_steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)

        # 数据送入设备
        input_msi = batch['input'].to(device)  # (B, C, H, W)
        gt_rgb = batch['gt_rgb'].to(device)  # (B, 3, H, W)
        gt_L = batch['gt_L'].to(device)  # (B, C)

        B, _, H, W = input_msi.shape

        # === 核心物理公式实现 (Eq. 4) ===
        # 1. 变换维度: (B, H, W, C)
        raw = input_msi.permute(0, 2, 3, 1)

        # 2. RGB Subspace Projection: I_MSI @ T_RGB
        proj_rgb = torch.matmul(raw, T_RGB)

        # 3. White Balancing: W = L @ T_RGB
        L_vec = gt_L.view(B, 1, 1, C)
        white_point = torch.matmul(L_vec, T_RGB)  # (B, 1, 1, 3)

        # Avoid division by zero
        wb_rgb = proj_rgb / (white_point + 1e-8)

        # 4. Color Correction: pred = wb @ M
        pred_linear = torch.matmul(wb_rgb, M)

        # 5. Gamma Correction (To sRGB)
        pred_srgb = torch.clamp(pred_linear, 0, 1) ** (1 / 2.2)

        # === Loss 计算 ===
        # Reconstruct Loss
        loss_rec = torch.nn.functional.mse_loss(pred_srgb.permute(0, 3, 1, 2), gt_rgb)

        # M Regularization (行和为1)
        row_sums = torch.sum(M, dim=1)
        loss_reg = torch.mean((row_sums - 1.0) ** 2)

        loss = loss_rec + 0.1 * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"   Step [{step}/{num_steps}] Loss: {loss.item():.6f} (Rec: {loss_rec.item():.6f})")

    print("\n✅ 光谱先验优化完成！")

    # --- 3. 保存结果到硬盘 ---
    os.makedirs(save_dir, exist_ok=True)
    # Detach detach并转到CPU保存
    t_final = T_RGB.detach().cpu()
    m_final = M.detach().cpu()

    torch.save({
        't_rgb': t_final,
        'm_static': m_final
    }, save_path)
    print(f"💾 参数已从 GPU 转移并保存至: {save_path}")

    return t_final, m_final


if __name__ == "__main__":
    # 如果你想单独测试这个文件，也可以直接运行它
    from config import SystemConfig

    cfg = SystemConfig()
    try:
        run_stage_1_optimization(cfg, "./data/KAIST")
    except Exception as e:
        print(f"测试运行需确保数据路径正确: {e}")