import torch
import os
import numpy as np
from config import SystemConfig
from pipeline import MSIReproductionPipeline
from utils_io import load_raw_file, save_srgb_image

# ==========================================
#      【在此处修改你的运行配置】
# ==========================================
# 1. 输入文件路径 (确保文件存在)
INPUT_PATH = "data/raw/sample_scene.raw"

# 2. 输出保存位置
OUTPUT_PATH = "data/output/result.png"

# 3. 权重路径 (如果有训练好的 .pth 文件，填路径；没有填 None)
# 例如: WEIGHTS_PATH = "checkpoints/best_model.pth"
WEIGHTS_PATH = None

# 4. 图像参数 (必须修改为与你的 .raw 文件匹配)
IMG_WIDTH = 480
IMG_HEIGHT = 300
IMG_CHANNELS = 9
IMG_BIT_DEPTH = 12


# ==========================================

def run():
    # 1. 打印信息
    print(f"--- MSI Color Reproduction Inference ---")
    print(f"输入: {INPUT_PATH}")
    print(f"输出: {OUTPUT_PATH}")

    # 2. 检查文件是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件: {INPUT_PATH}")
        # 为了防止报错，这里生成一个假文件演示流程 (可选)
        print("⚠️ 正在生成随机假数据用于演示流程...")
        os.makedirs(os.path.dirname(INPUT_PATH), exist_ok=True)
        dummy = np.random.randint(0, 2 ** IMG_BIT_DEPTH, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint16)
        dummy.tofile(INPUT_PATH)
        print("✅ 假数据已生成。")

    # 3. 加载配置
    config = SystemConfig()
    # 覆盖配置中的参数
    config.camera.input_width = IMG_WIDTH
    config.camera.input_height = IMG_HEIGHT
    config.camera.num_channels = IMG_CHANNELS
    config.camera.bit_depth = IMG_BIT_DEPTH

    # 4. 准备设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    # 5. 初始化模型
    model = MSIReproductionPipeline(config).to(device)
    model.eval()

    # 加载权重
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        print(f"加载权重: {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        print("⚠️ 未提供权重文件，使用随机初始化参数 (输出图片颜色将不正确)")

    # 6. 读取数据
    try:
        input_tensor = load_raw_file(
            INPUT_PATH,
            IMG_WIDTH,
            IMG_HEIGHT,
            IMG_CHANNELS,
            IMG_BIT_DEPTH
        ).to(device)
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    # 7. 推理
    with torch.no_grad():
        srgb_out, _ = model(input_tensor)

    # 8. 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_srgb_image(srgb_out, OUTPUT_PATH)
    print("✅ 推理完成，程序结束。")


if __name__ == "__main__":
    # 在 PyCharm 中直接右键运行此文件即可
    run()