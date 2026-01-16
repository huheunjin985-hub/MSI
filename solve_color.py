import torch
import os
import numpy as np
import cv2
import itertools
from config import SystemConfig
from models import MSIReproductionPipeline
from utils import load_raw_file

# ================= 配置区 =================
INPUT_PATH = "data/raw/sample_scene.raw"
WEIGHTS_PATH = "checkpoints/best_model.pth"  # 确保路径正确
OUTPUT_DIR = "data/output/color_debug"  # 结果保存到这个新文件夹
IMG_WIDTH = 480
IMG_HEIGHT = 300
IMG_CHANNELS = 9
IMG_BIT_DEPTH = 12


# ==========================================

def run():
    print("🚀 开始全通道颜色排查...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 准备模型和数据
    config = SystemConfig()
    config.camera.input_width = IMG_WIDTH
    config.camera.input_height = IMG_HEIGHT
    config.camera.num_channels = IMG_CHANNELS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSIReproductionPipeline(config).to(device)
    model.eval()
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        print(f"❌ 找不到权重文件: {WEIGHTS_PATH}")
        # 如果找不到，尝试用上一级目录的
        alt_path = "../../checkpoints/best_model.pth"
        if os.path.exists(alt_path):
            model.load_state_dict(torch.load(alt_path, map_location=device))
            print(f"✅ 已加载备用路径权重: {alt_path}")
        else:
            return

    # 2. 推理
    input_tensor = load_raw_file(INPUT_PATH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, IMG_BIT_DEPTH).to(device)
    with torch.no_grad():
        srgb_out, _ = model(input_tensor)

    # 3. 数据预处理 (取出 PyTorch Tensor -> Numpy)
    # 形状: (H, W, 3)
    original_img = srgb_out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # *关键步骤*: 如果图是竖着拼的，只切取中间部分 (预测图)
    h, w, c = original_img.shape
    if h > w:
        print(f"ℹ️ 检测到竖拼图 ({h}x{w})，正在切取中间部分...")
        single_h = h // 3
        # 取中间 1/3
        original_img = original_img[single_h: single_h * 2, :, :]

    # 归一化防止数值溢出
    original_img = np.clip(original_img, 0, 1)

    # 4. 暴力穷举 6 种 RGB 组合
    # 0=原通道1, 1=原通道2, 2=原通道3
    channels = [0, 1, 2]
    # 全排列: (0,1,2), (0,2,1), (1,0,2)... 共6种
    permutations = list(itertools.permutations(channels))

    # 颜色模式名称映射
    perm_names = {
        (0, 1, 2): "RGB_Original",
        (0, 2, 1): "RBG",
        (1, 0, 2): "GRB",
        (1, 2, 0): "GBR",
        (2, 0, 1): "BRG",
        (2, 1, 0): "BGR_Swapped"  # 这是我们要重点关注的
    }

    print("\n📸 正在生成 6 种颜色组合...")

    for perm in permutations:
        name = perm_names.get(perm, "Unknown")
        print(f"  -> 处理模式: {name} (顺序: {perm})")

        # A. 重新排列通道
        img_perm = original_img[:, :, perm]

        # B. 简单的 Gamma 校正 (提亮)
        # 用 1.8 比较适中，既不会太白也不会太黑
        img_gamma = np.power(img_perm, 1.0 / 1.8)

        # C. 转为 8-bit 用于保存
        img_save = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)

        # D. **OpenCV是BGR顺序**:
        # 为了让保存的图片所见即所得，我们需要再把 RGB 转回 BGR 给 OpenCV 保存
        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)

        filename = f"{OUTPUT_DIR}/Option_{name}.png"
        cv2.imwrite(filename, img_save)

    print(f"\n✅ 全部完成！请打开文件夹 '{OUTPUT_DIR}' 查看这 6 张图。")
    print("👉 挑选那张【人脸是肤色，字是黄色】的图片，记住它的文件名（例如 Option_BGR_Swapped.png）。")


if __name__ == "__main__":
    run()