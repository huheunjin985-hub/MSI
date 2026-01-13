###################################
# 测试代码
###################################
import numpy as np
import cv2
import os

# ================= 配置区域 (请确认这里) =================
FILE_PATH = "data/raw/sample_scene.raw"  # 你的 raw 文件路径

# 根据之前的测试，这些数值应该是对的
# 如果你的 config.py 里 width 是 1440, height 是 300，就填这里
# 注意：这里请填那个“大”的宽
WIDTH = 480
HEIGHT = 300
CHANNELS = 9


# =======================================================

def process_image():
    print(f"1. 正在读取文件: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print("❌ 错误: 找不到文件")
        return

    # === 读取数据 ===
    raw_data = np.fromfile(FILE_PATH, dtype=np.uint16)

    # === 步骤 A: 重塑形状 (HWC 模式) ===
    # 这是我们确定正确的模式，不要动
    try:
        img = raw_data.reshape((HEIGHT, WIDTH, CHANNELS))
    except ValueError:
        print(f"❌ 尺寸错误: 文件包含 {raw_data.size} 像素，但 {WIDTH}x{HEIGHT}x{CHANNELS} = {WIDTH * HEIGHT * CHANNELS}")
        return

    # === 步骤 B: 强制旋转 (解决“横着”的问题) ===
    # k=1 代表逆时针旋转90度。如果想要顺时针，改成 k=-1
    print("2. 正在执行旋转...")
    img = np.rot90(img, k=1, axes=(0, 1))

    # 旋转后，原本的宽变成了高，原本的高变成了宽
    # 现在 img.shape 应该是 (300, 480, 9) -> 变成 (480, 300, 9)
    new_h, new_w, _ = img.shape

    # === 步骤 C: 提取 RGB 并提亮 ===
    # 取前3个通道
    vis_img = img[:, :, 0:3].astype(np.float32)

    # 自动曝光 (找到最亮的值作为上限，防止全黑)
    max_val = np.percentile(vis_img, 99.9)
    if max_val == 0: max_val = 1
    vis_img = vis_img / max_val

    # Gamma 校正 (防止太暗)
    vis_img = np.power(vis_img, 1 / 2.2)

    # 转为 0-255
    vis_img = np.clip(vis_img * 255, 0, 255).astype(np.uint8)

    # 转为 OpenCV 的 BGR 格式
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # 保存旋转后的完整图
    cv2.imwrite("result_rotated.png", vis_img)
    print("✅ 已保存: result_rotated.png (这张图应该是竖着的了！)")

    # === 步骤 D: 尝试把三张并排的图切开 (解决“三张图”的问题) ===
    # 既然你说有三张图，说明这是一种 Side-by-Side 的拼接格式
    # 我们尝试把它们横向切成三份

    # 现在的宽度 new_w
    part_w = new_w // 3  # 假设三等分

    print(f"3. 正在尝试裁剪... 假设单张宽度为 {part_w}")
    if part_w > 10:  # 保护一下防止太小
        part_1 = vis_img[:, 0:part_w, :]  # 左边
        part_2 = vis_img[:, part_w:part_w * 2, :]  # 中间
        part_3 = vis_img[:, part_w * 2:part_w * 3, :]  # 右边

        cv2.imwrite("result_part_1.png", part_1)
        cv2.imwrite("result_part_2.png", part_2)
        cv2.imwrite("result_part_3.png", part_3)
        print("✅ 已尝试把图片切分成三份：result_part_1/2/3.png")


if __name__ == "__main__":
    process_image()