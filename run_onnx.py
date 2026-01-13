import onnxruntime as ort
import numpy as np
from utils_io import load_raw_file, save_srgb_image
from config import SystemConfig


def infer_with_onnx(onnx_path, raw_path):
    config = SystemConfig()

    # 1. 创建推理会话 (类似启动一个虚拟机)
    ort_session = ort.InferenceSession(onnx_path)

    # 2. 准备数据 (转为 Numpy，不需要 PyTorch)
    # 注意：load_raw_file 返回的是 Tensor，这里我们要转 numpy
    input_tensor = load_raw_file(raw_path, config.camera.input_width,
                                 config.camera.input_height, config.camera.num_channels)
    input_numpy = input_tensor.numpy()

    # 3. 运行推理
    inputs = {ort_session.get_inputs()[0].name: input_numpy}
    outputs = ort_session.run(None, inputs)

    # 4. 获取结果 (outputs[0] 对应 output_srgb)
    srgb_result = outputs[0]

    # 5. 保存
    # 需要将 numpy 转回 tensor 给 save_srgb_image 用，或者修改 save 函数
    import torch
    save_srgb_image(torch.from_numpy(srgb_result), "data/output/onnx_result.png")


if __name__ == "__main__":
    infer_with_onnx("msi_model.onnx", "data/raw/scene_001.raw")