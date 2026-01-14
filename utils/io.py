#######################################
# 负责“搬运数据”
# 只负责把数据从硬盘读到内存，或者从内存保存到硬盘，不负责修改数据内容。
#######################################

import numpy as np
import torch
import cv2
import os


def load_raw_file(file_path, width, height, channels, bit_depth=16):
    """
    Args:
        file_path: .raw 文件路径
        width: 宽 (请注意：如果是三张并排，这里应该是总宽度)
        height: 高
        channels: 通道数
        bit_depth: 位深
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    # 1. 确定位深
    if bit_depth <= 8:
        dtype = np.uint8
        max_val = 255.0
    else:
        # 10, 12, 16bit 统一按 uint16 读取
        dtype = np.uint16
        max_val = (2 ** bit_depth) - 1.0

    # 2. 读取二进制数据
    try:
        raw_data = np.fromfile(file_path, dtype=dtype)
    except Exception as e:
        raise IOError(f"Failed to read binary file: {e}")

    # 3. 尺寸校验
    expected_size = width * height * channels
    if raw_data.size != expected_size:
        raise ValueError(
            f"文件大小 ({raw_data.size}) 与配置 "
            f"W:{width}xH:{height}xC:{channels} ({expected_size}) 不匹配。\n"
            "建议：如果显示三张并排图，说明 Width 包含了所有波段的宽度，\n"
            "请确保 config.py 中的 input_width 是那个比较大的数字。"
        )

    # 4. 重塑形状 (HWC 模式 - 验证通过)
    # 既然之前的 A图 (HWC) 能看到清晰的车，说明这个模式是对的
    try:
        raw_img = raw_data.reshape((height, width, channels))
    except ValueError:
        # 备用方案：万一上面的失败，尝试 CHW
        raw_img = raw_data.reshape((channels, height, width))
        raw_img = raw_img.transpose(1, 2, 0)

    # 5. 转为浮点并归一化 (Numpy操作)
    raw_img = raw_img.astype(np.float32)

    # 6. 转 PyTorch Tensor
    tensor = torch.from_numpy(raw_img)
    tensor = tensor / max_val

    # 7. 维度调整: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    # ========================================================
    # 【新增】 自动旋转修复
    # 现象：车是横着的 (Width > Height 且车头朝左/右)
    # 动作：逆时针旋转 90 度，让车立起来
    # ========================================================
    # k=1: 逆时针90度 (Left -> Top)
    # k=-1: 顺时针90度 (Top -> Right)
    # dims=[2, 3] 是指在 H 和 W 维度上旋转
    tensor = torch.rot90(tensor, k=1, dims=[2, 3])

    return tensor


def save_srgb_image(tensor, output_path):
    """
    保存可视化图片
    """
    # 输入 tensor: (1, 3, H, W) 或 (1, C, H, W)
    # 我们只取前3个通道可视化
    if tensor.shape[1] > 3:
        # 如果通道数大于3，说明是多光谱，可视化前3个通道
        # 这就是为什么你会看到绿/蓝/黄三辆车的原因：它把不同波段画在了一起
        vis_tensor = tensor[:, 0:3, :, :]
    else:
        vis_tensor = tensor

    img = vis_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # 增强亮度 (自动曝光)
    # 你的图之前很黑，这里自动把它拉亮
    max_val = np.percentile(img, 99) if np.max(img) > 0 else 1.0
    img = img / max_val

    # Gamma 校正 (让暗部细节由于)
    img = np.power(img, 1 / 2.2)

    # 限制范围 0-255
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # RGB -> BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, img_bgr)
    print(f"✅ 图像已保存至: {output_path}")