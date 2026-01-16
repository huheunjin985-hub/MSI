

from .data_synthesis import SyntheticSpectralDataset
from .data_synthetic import generate_synthetic_data

# 对外暴露统一接口，方便上层调用
__all__ = ["SyntheticSpectralDataset", "generate_synthetic_data"]