import torch
import os
import sys  # 🔥 新增 sys 用于报错退出
import numpy as np
from config import SystemConfig
from models import MSIReproductionPipeline
from utils import load_raw_file, save_srgb_image  # 假设其他工具函数不需要变

# ==========================================
#      【在此处修改你的运行配置】
# ==========================================
# 1. 输入文件路径
INPUT_PATH = "data/raw/sample_scene.raw"

# 2. 输出保存位置
OUTPUT_PATH = "data/output/result_gpu_fixed.png"

# 3. 权重路径
WEIGHTS_PATH = "checkpoints/best_model.pth"

# 4. 图像参数
IMG_WIDTH = 480
IMG_HEIGHT = 300
IMG_CHANNELS = 9
IMG_BIT_DEPTH = 12


# ==========================================

def run():
    print(f"--- MSI Color Reproduction Inference (GPU Mode) ---")

    # 1. 强制检查 GPU (🔥 修改点 1)
    if not torch.cuda.is_available():
        print("❌ 严重错误: 未检测到 GPU！此代码强制要求 CUDA 环境。")
        sys.exit(1)

    # 强制锁定 GPU
    device = torch.device("cuda")
    print(f"✅ 已锁定设备: {torch.cuda.get_device_name(0)}")

    # 2. 检查输入文件
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件: {INPUT_PATH}")
        return  # 建议直接返回，不要再生成假数据干扰判断了

    # 3. 加载配置
    config = SystemConfig()
    config.camera.input_width = IMG_WIDTH
    config.camera.input_height = IMG_HEIGHT
    config.camera.num_channels = IMG_CHANNELS
    config.camera.bit_depth = IMG_BIT_DEPTH
    config.device = "cuda"  # 🔥 显式同步配置

    # 4. 初始化模型并上卡
    print("⏳ 正在加载模型...")
    model = MSIReproductionPipeline(config).to(device)
    model.eval()  # 开启评估模式

    # 5. 加载权重
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        # map_location 确保权重直接加载到 GPU
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ 成功加载权重: {WEIGHTS_PATH}")
    else:
        print("⚠️ 警告: 未找到训练好的权重文件，输出将是随机噪声！")

    # 6. 读取数据 (确保 .to(device))
    try:
        input_tensor = load_raw_file(
            INPUT_PATH,
            IMG_WIDTH,
            IMG_HEIGHT,
            IMG_CHANNELS,
            IMG_BIT_DEPTH
        ).to(device)  # 🔥 数据直接上 GPU

        # 增加一个维度 (Batch Size) 如果 load_raw_file 没有加的话
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    # 7. 推理 (开启 no_grad 节省显存)
    print("🚀 正在推理...")
    with torch.no_grad():
        srgb_out, _ = model(input_tensor)

        # ==========================================================
        #  【关键修复步骤】
        # ==========================================================
        print(">>> 正在应用色彩修复...")

        # 🔥 修改点 2: 修正通道顺序 BGR -> RGB
        # 原代码 [0, 1, 2] 是没变的，必须改成 [2, 1, 0] 才能交换红蓝通道
        # srgb_out = srgb_out[:, [2, 1, 0], :, :]
        # print("  ✅ 已执行 BGR -> RGB 通道交换")

        # 🔥 修改点 3: Gamma 校正与截断
        # 先限制范围在 0-1 之间，防止负数导致 pow 报错
        srgb_out = torch.clamp(srgb_out, 0.0, 1.0)

        # Gamma 2.2 校正 (Linear -> sRGB)
        # 如果出来的图片很白/雾蒙蒙，说明模型已经学到了Gamma，请注释掉下面这行
        srgb_out = torch.pow(srgb_out, 1.0 / 2.2)
        print("  ✅ Gamma (2.2) 校正已应用")
        # ==========================================================

    # 8. 保存
    # 如果 save_srgb_image 函数里包含 .cpu() 转换，这里就不用管；
    # 如果报错，可能需要先 srgb_out.cpu()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_srgb_image(srgb_out, OUTPUT_PATH)  # 确保 utils 里处理了 tensor

    print(f"✅ 保存成功: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()