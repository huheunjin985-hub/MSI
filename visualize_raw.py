import numpy as np
import cv2
import os

# === 请确保这里的宽、高是 input_width / input_height ===
# 根据之前的报错，您的图片可能是 480x300 或类似的比例
# 如果您不知道具体数值，请保持您 config.py 里原本的数值
WIDTH = 480  # 请核对您的真实宽度
HEIGHT = 300  # 请核对您的真实高度
CHANNELS = 9  # 通道数

INPUT_PATH = "data/raw/sample_scene.raw"  # 修改为您的文件路径


def fix_image():
    # 1. 读取数据
    raw_data = np.fromfile(INPUT_PATH, dtype=np.uint16)

    # 2. 形状重塑 (回到之前能看到车的那个模式)
    # 这种 "H, W, C" 模式是最常用的 Interleaved 格式
    try:
        img = raw_data.reshape((HEIGHT, WIDTH, CHANNELS))
    except ValueError:
        # 如果尺寸报错，说明宽/高可能填反了，自动交换尝试
        print("⚠️ 尺寸不匹配，尝试交换宽高...")
        img = raw_data.reshape((WIDTH, HEIGHT, CHANNELS))

    # 3. 解决“全红”问题 (颜色通道选择)
    # 之前的红色是因为默认取了 [0,1,2] 且通道0是红外或高响应波段
    # 我们尝试换一组通道，比如 [2, 1, 0] (BGR) 或者 [4, 2, 0]
    # 这里先试 [2, 1, 0]，通常能把红色变成正常的蓝/灰色
    vis_img = img[:, :, [2, 1, 0]].astype(np.float32)

    # 4. 解决“三张叠在一起”的问题 (裁剪)
    # 如果原图是9通道，您之前看到叠在一起可能是因为误把所有通道当成高了
    # 但我们现在只取了 3 个通道，应该只会在一张图里
    # 如果显示出来还是叠着的，那就在这里切片：
    # h, w, _ = vis_img.shape
    # vis_img = vis_img[0:h//3, :, :] # 只取上面 1/3

    # 5. 亮度增强 (归一化 + Gamma)
    # 避免全黑或全白
    p99 = np.percentile(vis_img, 98)
    if p99 > 0:
        vis_img = vis_img / p99
    vis_img = np.power(vis_img, 1.0 / 2.2)  # 变亮
    vis_img = np.clip(vis_img * 255, 0, 255).astype(np.uint8)

    # 6. 解决方向和镜像 (终极修正)
    # 如果之前车是竖着的，这里不要 rot90，保持原样看看
    # 如果发现是倒着的，可以取消下面注释：
    # vis_img = np.rot90(vis_img, k=1)
    def fix_orientation(img_array):
        """
        解决图像“横着”的问题，逆时针旋转90度
        img_array: (H, W, C)
        """
        # 对应 final_fix.py 中的 np.rot90(..., k=1)
        return np.rot90(img_array, k=1, axes=(0, 1))

    # 必须做的：解决文字镜像
    vis_img = np.flip(vis_img, axis=1)

    # 7. 保存
    cv2.imwrite("fixed_visualization.png", vis_img)
    print("✅ 处理完成！已保存 fixed_visualization.png")
    print("提示：如果颜色偏蓝/偏绿，请修改代码第28行的 [2, 1, 0] 顺序")


if __name__ == "__main__":
    if os.path.exists(INPUT_PATH):
        fix_image()
    else:
        print("找不到文件")