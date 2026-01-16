##################################################
#   查看.h5文件内是什么
##################################################
import h5py
import numpy as np
import os

# === 请修改这里的路径为您的任意一个 .h5 文件路径 ===
FILE_PATH = "D:/多光谱/数据集/KAUST_SpectralReflectanceImages_h5/h5/2019-08-23_002.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"📄 数据集(Dataset): '{name}' | 形状: {obj.shape} | 类型: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Tk 文件夹(Group): '{name}'")

if not os.path.exists(FILE_PATH):
    print(f"❌ 找不到文件: {FILE_PATH}")
else:
    print(f"正在检查: {FILE_PATH} ...")
    with h5py.File(FILE_PATH, 'r') as f:
        f.visititems(print_structure)