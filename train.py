# ⚠️ 新增: 加载外部的光谱先验优化模块 (原 `train_optimize.py` 中的函数)
#################################################################
# Rec:RGB图像的MSE（值越小越好）。通常认为 <0.05 就是肉眼看不出差异的水平
# Ang:光照估计的角度误差（单位为度）。
# Loss:
################################################################
from train_optimize import run_stage_1_optimization

import torch
import os
import sys
from torch.utils.data import DataLoader
from config import SystemConfig
from models.pipeline import MSIReproductionPipeline
from datasets.data_synthesis import SyntheticSpectralDataset
from models.losses import PaperLoss


def train():
    # ==========================
    # 1. 基础配置与设备初始化
    # ==========================
    config = SystemConfig()
    # 读取配置文件中的设备参数，避免硬编码
    # 强制检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ 错误: 代码要求必须在 GPU 上运行，但未检测到 CUDA 设备！")
        sys.exit(1)  # 直接退出程序

    # 强制指定设备为 cuda
    device = torch.device("cuda")
    print(f"✅ 已锁定 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存状态: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB used")
    print(f"📋当前配置: Batch={config.model.batch_size}, LR={config.model.lr}, Epochs={config.model.max_epochs}")

    # # ==========================
    # # 2. 数据准备
    # # ==========================
    # HSI_DATA_PATH = config.dataset.train_data_root  # 从配置读取路径，而非硬编码
    # if not os.path.exists(HSI_DATA_PATH) or len(os.listdir(HSI_DATA_PATH)) == 0:
    #     print(f"❌ 错误: 在 {HSI_DATA_PATH} 找不到数据集！")
    #     print("💡 请将 .h5 文件放入指定文件夹，或修改 config.py 中的 train_data_root")
    #     return
    #
    # print(f"📂 正在加载数据集: {HSI_DATA_PATH}")
    # dataset = SyntheticSpectralDataset(hsi_data_root=HSI_DATA_PATH, config=config)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=config.model.batch_size,  # 从配置读取 batch_size
    #     shuffle=True,
    #     num_workers=os.cpu_count() // 2,  # 自动根据CPU核心数设置 workers
    #     pin_memory=True if torch.cuda.is_available() else False  # 加速GPU传输
    # )
    # print(f"✅ 数据加载完毕，预计每个 epoch 有 {len(dataloader)} 个 Batch")
    # ==========================
    # 2. 数据准备
    # ==========================
    # ✅ 修改点：从 config.dataset 中读取路径
    hsi_data_root = config.dataset.train_data_root

    # 简单的路径检查
    if not os.path.exists(hsi_data_root):
        print(f"❌ 错误: 数据集路径不存在 -> {hsi_data_root}")
        print("💡 请在 config.py 中修改 dataset.train_data_root")
        return

    print(f"📂 正在加载数据集: {hsi_data_root}")
    dataset = SyntheticSpectralDataset(hsi_data_root=hsi_data_root, config=config)

    # ✅ 修改点：使用 config.num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
        num_workers=config.num_workers,  # 统一由 config 控制
        pin_memory=(config.device == "cuda")  # GPU下开启 pin_memory
    )
    print(f"✅ 数据加载完毕，共 {len(dataloader)} 个 Batch")

    # ==========================
    # 3. 论文 Stage 1: 光谱先验优化
    # ==========================
    # 调用外部模块实现的 Algorithm 1
    T_init, M_init = run_stage_1_optimization(config, hsi_data_root)

    # ==========================
    # 4. 模型与优化器 (论文 Stage 2)
    # ==========================
    model = MSIReproductionPipeline(
        config,
        initial_T=T_init,  # 注入 Stage 1 的物理先验
        initial_M=M_init
    ).to(device)

    model.train()  # 开启训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.lr)  # 从配置读取学习率
    criterion = PaperLoss(
        lambda_ae=config.model.lambda_ae,  # 从配置读取损失权重 (论文 Eq.6)
        lambda_rmse=config.model.lambda_rmse  # 从配置读取损失权重 (论文 Eq.8)
    ).to(device)
    print(f"🔧 已初始化模型: {model.__class__.__name__}, 损失函数: {criterion.__class__.__name__}")

    # ==========================
    # 5. 训练循环 (Training Loop)
    # ==========================
    epochs = config.model.max_epochs  # 从配置读取训练轮数
    # 提前创建checkpoint目录
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n🏁 开始训练，共 {epochs} 个 Epoch")
    for epoch in range(epochs):
        epoch_loss = 0.0
        # 重置 batch 损失统计
        total_rec_loss = 0.0
        total_ang_loss = 0.0

        for i, batch in enumerate(dataloader):
            # 数据送入设备
            raw_msi = batch['input'].to(device)
            gt_rgb = batch['gt_rgb'].to(device)
            gt_L = batch['gt_L'].to(device)

            optimizer.zero_grad()
            # 前向传播，与模型架构对齐
            pred_rgb, intermediates = model(raw_msi)

            # 计算损失 (与论文公式 (6)(7)(8) 完全对齐)
            l_hat = intermediates['l_hat']
            loss, rec_val, ang_val = criterion(pred_rgb, gt_rgb, l_hat, gt_L)

            loss.backward()
            optimizer.step()

            # 累积损失数据
            epoch_loss += loss.item()
            total_rec_loss += rec_val
            total_ang_loss += ang_val

            # 每 10 个 batch 打印一次进度
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{i}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} (Rec: {rec_val:.4f}, Ang: {ang_val:.2f}°)"
                )

        # 每个 Epoch 结束后的统计与保存
        avg_loss = epoch_loss / len(dataloader)
        avg_rec = total_rec_loss / len(dataloader)
        avg_ang = total_ang_loss / len(dataloader)

        print(
            f"==== Epoch {epoch + 1} 结束 | "
            f"平均 Loss: {avg_loss:.4f}, 平均 RecLoss: {avg_rec:.4f}, 平均 AngLoss: {avg_ang:.2f}° ===="
        )

        # 保存 checkpoint，根据配置的频率
        if (epoch + 1) % config.model.save_freq == 0:
            save_path = f"checkpoints/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 权重已保存至: {save_path}")

    # 保存最终模型
    final_path = "checkpoints/best_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n🎉 训练全部完成！最终模型保存至: {final_path}")
    print(f"👉 现在可以运行 run_inference.py 测试效果啦！")


if __name__ == "__main__":
    train()