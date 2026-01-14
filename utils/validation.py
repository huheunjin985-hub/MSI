###########################################################
# 负责“修理数据”
############################################################
# utils/validation.py
import numpy as np
import cv2


def fix_orientation(img_array):
    """
    解决图像“横着”的问题，逆时针旋转90度
    img_array: (H, W, C)
    """
    # 对应 final_fix.py 中的 np.rot90(..., k=1)
    return np.rot90(img_array, k=0, axes=(0, 1))


def split_side_by_side(img_array, num_parts=3):
    """
    解决图像“三张并排”的问题，切分成列表
    img_array: (H, W, C)
    返回: [img1, img2, img3]
    """
    h, w, c = img_array.shape
    part_w = w // num_parts

    parts = []
    for i in range(num_parts):
        # 切片操作：img[:, start:end, :]
        part = img_array[:, i * part_w: (i + 1) * part_w, :]
        parts.append(part)
    return parts


def auto_enhance(img_array):
    """
    自动曝光和Gamma校正，用于可视化验证
    img_array: (H, W, 3) 必须是RGB通道
    """
    vis_img = img_array.astype(np.float32)
    # 自动曝光
    max_val = np.percentile(vis_img, 99.9)
    if max_val == 0: max_val = 1
    vis_img = vis_img / max_val
    # Gamma校正
    vis_img = np.power(vis_img, 1 / 2.2)
    # 转回uint8
    return np.clip(vis_img * 255, 0, 255).astype(np.uint8)