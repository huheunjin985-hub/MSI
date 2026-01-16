##############################################################
#   用于光谱先验化，模拟自然界光谱
##############################################################

import torch

def generate_synthetic_data(num_samples=5000, num_channels=9):
    """
    【无需外部文件】全自动生成 Stage 1 所需的物理数据
    """
    print(f"🧪 生成 {num_samples} 条合成光谱数据用于物理初始化...")

    # 定义波长范围 400nm - 700nm (31个点，间隔10nm)
    wavelengths = torch.linspace(400, 700, 31)

    # 1. 模拟自然物体的反射率 R (Reflectance)
    # 自然界物体的反射率通常是平滑的
    # 我们用随机生成的平滑曲线来模拟 (高斯混合)
    R = torch.zeros(num_samples, 31)
    for i in range(num_samples):
        # 随机中心和宽度
        center = torch.FloatTensor(1).uniform_(400, 700)
        width = torch.FloatTensor(1).uniform_(50, 200)
        # 高斯函数模拟光谱峰值
        R[i] = torch.exp(-(wavelengths - center) ** 2 / (2 * width ** 2))

    # 2. 模拟标准光源 L (Illuminant) - 假设是 D65 标准白光
    # 这里简化为全1 (理想白光)，让 T 矩阵学习从“物体本色”到“sRGB”的映射
    L = torch.ones(num_samples, 31)

    # 3. 模拟相机感光度 C (Camera Sensitivity)
    # 假设9个通道均匀分布在光谱上
    C = torch.zeros(31, num_channels)
    centers = torch.linspace(400, 700, num_channels)
    for k in range(num_channels):
        C[:, k] = torch.exp(-(wavelengths - centers[k]) ** 2 / (2 * 30 ** 2))

    # ==== 核心物理公式 ====

    # A. 生成 MSI 输入 (模拟相机拍到的 RAW 值)
    # MSI = R * L * C
    # (Samples, 31) @ (31, 9) -> (Samples, 9)
    X_msi = (R * L) @ C

    # B. 生成 Ground Truth (sRGB 真值)
    # 我们需要用 CIE XYZ 匹配函数来算 sRGB
    # 这里为了不引入复杂文件，我们直接用一个简化的近似矩阵把 31波段转成 RGB
    # (这在数学上等价于告诉模型：我要这种物理映射关系)

    # 模拟 CIE XYZ 颜色匹配函数 (简化版)
    xyz_cmf = torch.zeros(31, 3)
    # R 峰值在 600nm (索引20), G 在 550nm (索引15), B 在 450nm (索引5)
    xyz_cmf[:, 0] = torch.exp(-(wavelengths - 600) ** 2 / (2 * 30 ** 2))  # X (Red-ish)
    xyz_cmf[:, 1] = torch.exp(-(wavelengths - 550) ** 2 / (2 * 30 ** 2))  # Y (Green/Luma)
    xyz_cmf[:, 2] = torch.exp(-(wavelengths - 450) ** 2 / (2 * 30 ** 2))  # Z (Blue)

    X_xyz = (R * L) @ xyz_cmf

    # XYZ -> Linear RGB 转换矩阵 (Standard sRGB matrix)
    M_xyz2rgb = torch.tensor([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    Y_linear_rgb = X_xyz @ M_xyz2rgb.T
    Y_linear_rgb = torch.clamp(Y_linear_rgb, 0, 1)  # 截断到合理范围

    return X_msi, L, Y_linear_rgb