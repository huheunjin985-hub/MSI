#########################################
# 配置模块：定义了多光谱相机参数、模型参数和固定参数。
#########################################
# from dataclasses import dataclass, field
# from typing import List
#
# @dataclass
# class MSICameraConfig:
#     """
#     多光谱相机硬件参数配置
#     已修复：扁平化结构，确保属性可被直接访问
#     """
#     # === 您的自定义硬件参数 ===
#     num_channels: int = 9       # 您的相机通道数
#     wavelength_start: int = 400 # 起始波长
#     wavelength_end: int = 700   # 终止波长
#     bit_depth: int = 10         # 您的 RAW 位深 (10bit)
#
#     # 您的 RAW 文件分辨率
#     input_height: int = 480
#     input_width: int = 300
#
#     # 模拟的光谱灵敏度曲线路径 (如有真实数据需替换，这里给个默认值防止报错)
#     spectral_sensitivity_path: str = "./data/sensor_sensitivity.npy"
#
# @dataclass
# class ModelConfig:
#     """
#     模型训练相关参数
#     根据论文中提到的 CSPDNet 和 MLP 结构设定
#     """
#     backbone_type: str = "cspdnet"  # 论文提及使用 YOLOv4 的 CSPDNet
#     hidden_dim: int = 64            # MLP 隐藏层维度
#     pretrained_weights: str = None  # 预训练权重路径
#
#     # 损失函数权重 (参考论文 Eq 6, 7, 8)
#     lambda_ae: float = 1.0     # Angular Error 权重 (Illuminance)
#     lambda_rmse: float = 10.0  # RMSE 权重 (Matrix regression)
#
# @dataclass
# class FixedParams:
#     """
#     论文中提及的固定参数或优化后的固定矩阵
#     """
#     # 目标色彩空间 (Target RGB Space) 通常指 sRGB 或 CIE XYZ
#     target_space: str = "sRGB"
#
#     # 这里的矩阵通常通过 'Spectral Prior-guided Optimization' 阶段算出
#     # 预留接口，实际运行时会从文件加载
#     initial_trgb_path: str = "./checkpoints/T_rgb_init.npy"
#     initial_m_static_path: str = "./checkpoints/M_static_init.npy"
#
# @dataclass
# class SystemConfig:
#     camera: MSICameraConfig = field(default_factory=MSICameraConfig)
#     model: ModelConfig = field(default_factory=ModelConfig)
#     fixed: FixedParams = field(default_factory=FixedParams)
#
#     # 企业级部署参数
#     device: str = "cuda"   # 'cuda' or 'cpu'
#     precision: str = "fp32" # 'fp16' for speedup


from dataclasses import dataclass, field
import torch

@dataclass
class MSICameraConfig:
    """相机硬件参数配置"""
    num_channels: int = 9  # 通道数
    wavelength_start: int = 400  # 起始波长
    wavelength_end: int = 700  # 终止波长
    bit_depth: int = 12  # RAW位深
    input_height: int = 300  # 输入高度
    input_width: int = 480  # 输入宽度


@dataclass
class ModelConfig:
    """模型架构参数"""
    backbone_type: str = "cspdnet"
    hidden_dim: int = 64

    # 训练超参数
    batch_size: int = 16  # 批大小
    lr: float = 1e-4  # 学习率
    max_epochs: int = 250  # 总 Epoch 数
    save_freq: int = 5  # 保存频率 (多少个Epoch保存一次)
    # Loss 权重
    lambda_ae: float = 1.0  # 角度误差权重
    lambda_rmse: float = 10.0  # 矩阵回归权重


@dataclass
class FixedParams:
    """固定路径参数"""
    initial_trgb_path: str = "./checkpoints/priors/t_rgb_init.pth"
    initial_m_static_path: str = "./checkpoints/priors/m_static_init.pth"

@dataclass
class DatasetConfig:
    """数据集路径配置"""
    # 训练数据所在的根目录 (包含 .h5 或 .mat 文件)
    train_data_root: str = "./data/KAIST"
    # 测试数据路径 (可选)
    test_data_path: str = "./data/KAIST/test"


@dataclass
class SystemConfig:
    """系统总配置"""
    # 子模块配置
    camera: MSICameraConfig = field(default_factory=MSICameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fixed: FixedParams = field(default_factory=FixedParams)

    # ✅ 这里必须把 dataset 注册进来！
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    device: str = "cuda"  # 训练时建议用 cuda
    num_epochs: int = 2000
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 0  # (Windows下如果报错，可改为0)