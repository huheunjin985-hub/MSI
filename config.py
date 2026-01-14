#########################################
# 配置模块：定义了多光谱相机参数、模型参数和固定参数。
#########################################
from dataclasses import dataclass, field
from typing import List


@dataclass
class MSICameraConfig:
    """
    多光谱相机硬件参数配置
    根据市面上主流多光谱相机（如IMEC, Ximea等）设定
    """
    # num_channels: int = 16  # 主流通道数：16通道 (4x4 mosaic) 或 25通道 (5x5)
    # wavelength_start: int = 400  # 起始波长 (nm)
    # wavelength_end: int = 700  # 终止波长 (nm)
    # bit_depth: int = 12  # 原始RAW位深
    #
    # # 模拟的光谱灵敏度曲线路径 (如有真实数据需替换)
    # spectral_sensitivity_path: str = "./data/sensor_sensitivity.npy"
    #
    # # 图像分辨率 (用于预处理resize)
    # input_height: int = 512
    # input_width: int = 512

    @dataclass
    class MSICameraConfig:
        num_channels: int = 9  # 你的相机通道数 (例如 IMEC 4x4 是 16)
        wavelength_start: int = 400 #波长
        wavelength_end: int = 700 #波长
        bit_depth: int = 10  # 常见的 RAW 通常是 10 或 12 bit，存储为 16bit

        # 必须与你的 .raw 文件像素完全对应！
        input_height: int = 300  #
        input_width: int = 480  #


@dataclass
class ModelConfig:
    """
    模型训练相关参数
    根据论文中提到的CSPDNet和MLP结构设定
    """
    backbone_type: str = "cspdnet"  # 论文提及使用 YOLOv4 的 CSPDNet
    hidden_dim: int = 64  # MLP 隐藏层维度
    pretrained_weights: str = None  # 预训练权重路径

    # 损失函数权重 (参考论文 Eq 6, 7, 8)
    lambda_ae: float = 1.0  # Angular Error 权重 (Illuminance)
    lambda_rmse: float = 10.0  # RMSE 权重 (Matrix regression)


@dataclass
class FixedParams:
    """
    论文中提及的固定参数或优化后的固定矩阵
    """
    # 目标色彩空间 (Target RGB Space) 通常指 sRGB 或 CIE XYZ
    target_space: str = "sRGB"

    # 这里的矩阵通常通过 'Spectral Prior-guided Optimization' 阶段算出
    # 预留接口，实际运行时会从文件加载
    initial_trgb_path: str = "./checkpoints/T_rgb_init.npy"
    initial_m_static_path: str = "./checkpoints/M_static_init.npy"


@dataclass
class SystemConfig:
    camera: MSICameraConfig = field(default_factory=MSICameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fixed: FixedParams = field(default_factory=FixedParams)

    # 企业级部署参数
    device: str = "cuda"  # 'cuda' or 'cpu'
    precision: str = "fp32"  # 'fp16' for speedup

