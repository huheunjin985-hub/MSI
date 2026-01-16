"""
严格对应论文 Algorithm 2: Synthetic Data Generation
职责：
1. 读取高光谱图像 (Hyperspectral Image, HSI)
2. 随机采样光源 (Illuminant)
3. 模拟相机成像过程生成 Input(MSI) 和 GT(sRGB)
"""

import torch
import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from utils.cie_data import get_cie_xyz_31

class SyntheticSpectralDataset(Dataset):
    """
    针对您的 KAIST .h5 数据进行适配
    Feature: Key='img', Shape=(34, H, W)
    """

    def __init__(self, hsi_data_root, config, split='train'):
        self.config = config

        # 1. 扫描文件
        self.files = [
            os.path.join(hsi_data_root, f)
            for f in os.listdir(hsi_data_root)
            if f.endswith('.h5') or f.endswith('.hdf5')
        ]

        if len(self.files) == 0:
            print(f"⚠️ 警告: 在 {hsi_data_root} 下没找到 .h5 文件！")
        else:
            print(f"✅ 已加载数据集，共找到 {len(self.files)} 个场景。")

        # 2. 物理参数初始化 (标准 31 波段: 400nm-700nm)
        # 实际情况应该加载真实的 CIE 1931 曲线 csv
        # self.cie_xyz = torch.abs(torch.randn(31, 3))
        # self.cie_xyz = self.cie_xyz / self.cie_xyz.sum(dim=0)  # 归一化
        # ----------------------------------------------------
        # 【修改】使用真实的 CIE 物理数据，替代原来的随机生成
        # ----------------------------------------------------

        # 1. 物理 CIE XYZ 匹配函数 (标准真值)
        # 形状: (31, 3)
        self.cie_xyz = get_cie_xyz_31()
        # 归一化 (让 Y 通道的最大值为 1，符合光度学定义)
        self.cie_xyz = self.cie_xyz / self.cie_xyz[:, 1].sum()

        # 2. 相机光谱响应 (Camera Sensitivity)
        # 如果您没有真实的相机参数文件，
        # 用高斯模拟是目前最安全的方法 (比随机数好得多，因为物理响应是平滑的)
        print("📷 正在模拟高斯相机响应 (9通道)...")
        self.camera_sens = torch.zeros(31, 9)
        wavelengths = torch.linspace(400, 700, 31)
        # 假设 9 个通道均匀分布在 400-700nm 之间
        centers = torch.linspace(420, 680, 9)
        for k in range(9):
            # 标准差 20nm 左右
            self.camera_sens[:, k] = torch.exp(-(wavelengths - centers[k]) ** 2 / (2 * 25 ** 2))

        # 归一化相机响应
        self.camera_sens = self.camera_sens / self.camera_sens.max()
        # ----------------------------------------------------

    def __len__(self):
        # 数据增强：每个场景生成 4 种光照
        return len(self.files) * 4

    def __getitem__(self, idx):
        file_idx = idx // 4
        h5_path = self.files[file_idx]

        # 1. 读取数据
        try:
            with h5py.File(h5_path, 'r') as f:
                # === 针对您的数据修改 ===
                if 'img' in f:
                    hsi_data = f['img'][:]  # 读取数据
                else:
                    # 备用方案：读取任何第一个key
                    key = list(f.keys())[0]
                    hsi_data = f[key][:]

        except Exception as e:
            print(f"❌ 读取错误 {h5_path}: {e}")
            hsi_data = np.random.rand(34, 512, 512)  # 假数据防止崩溃

        # 2. 格式清理
        # 您的数据是 float64，需要转 float32
        hsi_data = hsi_data.astype(np.float32)

        # 您的数据是 (34, 512, 512) -> CHW 格式
        # 我们计算需要 HWC 格式 -> (512, 512, 34)
        if hsi_data.shape[0] < hsi_data.shape[1]:
            hsi_data = np.transpose(hsi_data, (1, 2, 0))

        # 3.不仅要 HWC，还要处理通道数 (34 -> 31)
        # 我们的物理矩阵是 (31, 3) 和 (31, 9)，所以输入必须是 31 通道
        # 通常 KAIST 的前 31 个通道就是可见光范围
        if hsi_data.shape[2] > 31:
            hsi_data = hsi_data[:, :, :31]
        elif hsi_data.shape[2] < 31:
            # 万一通道不够，抛出错误
            raise ValueError(f"数据通道数不足31: {hsi_data.shape}")

        # 4. 随机裁剪 (H, W, 31) -> (128, 128, 31)
        # 显存优化：如果不切，512x512可能会爆显存
        H, W, C = hsi_data.shape
        crop_size = 128

        if H > crop_size and W > crop_size:
            x = np.random.randint(0, W - crop_size)
            y = np.random.randint(0, H - crop_size)
            hsi_crop = hsi_data[y:y + crop_size, x:x + crop_size, :]
        else:
            hsi_crop = hsi_data

        # 转 Tensor
        radiance = torch.from_numpy(hsi_crop).float()
        # 归一化 (防止原始数值过大)
        if radiance.max() > 1.0:
            radiance = radiance / (radiance.max() + 1e-6)

        # 5. 模拟光照与成像 (核心物理过程)
        illuminant = torch.rand(31) + 0.1
        illuminant = illuminant / illuminant.max()

        # 场景反射 E = R * L
        scene_radiance = radiance * illuminant

        # 生成 GT RGB (31->3)
        gt_xyz = torch.matmul(scene_radiance, self.cie_xyz)
        gt_rgb = self.xyz_to_srgb(gt_xyz)

        # 生成 Input MSI (31->9)
        input_msi = torch.matmul(scene_radiance, self.camera_sens)

        # 生成 GT Illuminant (用于 Loss)
        gt_L = torch.matmul(illuminant, self.camera_sens)

        # 6. 输出 (Permute to CHW for PyTorch)
        return {
            'input': input_msi.permute(2, 0, 1),  # (9, 128, 128)
            'gt_rgb': gt_rgb.permute(2, 0, 1),  # (3, 128, 128)
            'gt_L': gt_L  # (9,)
        }

    def xyz_to_srgb(self, xyz):
        """
        严格按照论文复现：标准 sRGB 转换过程 (IEC 61966-2-1)
        包含：
        1. 高精度 XYZ -> Linear RGB 矩阵 (D65)
        2. 标准分段 Gamma 校正 (比简单的 **1/2.2 更准确，保护暗部细节)
        """
        # 1. XYZ -> Linear RGB 转换矩阵 (Standard sRGB D65)
        # 精度比原来的简写版更高
        M = torch.tensor([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ], dtype=torch.float32)

        # 矩阵乘法
        rgb_linear = torch.matmul(xyz, M.t())

        # 2. 截断范围 (Gamut Mapping)
        # 即使是物理数据，转换后也可能出现微小的负值或超过1的值，需截断
        rgb_linear = torch.clamp(rgb_linear, 0.0, 1.0)

        # 3. 标准 sRGB Gamma 校正 (Transfer Function)
        # 公式:
        # C_srgb = 12.92 * C_linear,                  if C_linear <= 0.0031308
        # C_srgb = 1.055 * C_linear^(1/2.4) - 0.055,  if C_linear >  0.0031308

        rgb = torch.where(
            rgb_linear <= 0.0031308,
            12.92 * rgb_linear,
            1.055 * torch.pow(rgb_linear, 1.0 / 2.4) - 0.055
        )

        return rgb