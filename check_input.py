#########################
# 测试代码
##############################



import numpy as np
import cv2
import os

# ================= 配置区域 =================
# 原图路径
FILE_PATH = "data/raw/sample_scene.raw"

# 如果之前的图片里没有明显的“斜纹/雪花”，说明宽高大概率是对的
# 这里保留你之前的设置，或者试着互换一下
WIDTH = 480  # 宽
HEIGHT = 300  # 高
CHANNELS = 9  # 通道数


# ===========================================

def normalize_for_display(img_data):
    """
    自动“夜视仪”模式：可以把极暗的 RAW 数据变亮
    """
    img_data = img_data.astype(np.float32)

    # 1. 自动色阶：找到最亮的像素(排除掉前1%的噪点)，作为最大值
    # 哪怕你的图很黑，这一步也会把它强行拉得非常亮
    max_val = np.percentile(img_data, 99.5)
    if max_val == 0: max_val = 1.0  # 防止除以0

    print(f"  -> debug: 原始数据最大值约为 {np.max(img_data)}, 归一化采用上限: {max_val}")

    img_data = img_data / max_val
    img_data = np.clip(img_data, 0, 1)  # 截断超过1的部分

    # 2. Gamma 校正 (提亮暗部)
    # RAW 图必须做这一步人眼才能看清细节
    img_data = np.power(img_data, 1 / 2.2)

    return (img_data * 255).astype(np.uint8)


def check_raw_bright():
    print(f"正在读取: {FILE_PATH}")
    print(f"设置分辨率: {WIDTH}x{HEIGHT}, 通道: {CHANNELS}")

    if not os.path.exists(FILE_PATH):
        print("❌ 文件不存在")
        return

    raw_data = np.fromfile(FILE_PATH, dtype=np.uint16)

    expected = WIDTH * HEIGHT * CHANNELS
    if raw_data.size != expected:
        print(f"❌ 尺寸对不上! 实际: {raw_data.size} ({raw_data.size / CHANNELS} 像素)")
        print(f"   配置: {expected}")
        return

    # === 方案 A: 测试 HWC (像素优先) ===
    try:
        img_hwc = raw_data.reshape((HEIGHT, WIDTH, CHANNELS))
        # 取前3个通道合成
        vis_a = img_hwc[:, :, 0:3]
        # 强制提亮
        vis_a = normalize_for_display(vis_a)
        cv2.imwrite("check_A_HWC_bright.png", vis_a)
        print("✅ 已生成 check_A_HWC_bright.png (如果此图正常，说明是 HWC 格式)")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    check_raw_bright()