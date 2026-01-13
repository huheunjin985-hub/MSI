import torch
from config import SystemConfig
from pipeline import MSIReproductionPipeline

def export():
    # 1. 加载配置和模型
    config = SystemConfig()
    model = MSIReproductionPipeline(config)
    model.eval() # 必须切换到评估模式

    # 2. 如果你有训练好的权重，必须在这里加载！
    # model.load_state_dict(torch.load("checkpoints/best_model.pth"))

    # 3. 创建一个虚拟输入 (Dummy Input) 用于追踪数据流
    # 格式: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, config.camera.num_channels,
                              config.camera.input_height,
                              config.camera.input_width)

    # 4. 导出
    output_path = "msi_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,        # 导出内部权重
        opset_version=11,          # 通用算子集版本
        do_constant_folding=True,  # 优化常量
        input_names=['input_msi'], # 指定输入节点名
        output_names=['output_srgb', 'l_vec', 'm_mat'], # 指定输出节点名
        dynamic_axes={'input_msi': {0: 'batch_size'}, 'output_srgb': {0: 'batch_size'}}
    )
    print(f"✅ 模型已成功导出为: {output_path}")

if __name__ == "__main__":
    export()