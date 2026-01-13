import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional


class MultispectralRAWReader:
    """
    多光谱RAW图像读取器
    完全模拟MATLAB的read_raw.m实现
    """

    def __init__(self, config: dict = None):
        """
        初始化读取器

        参数:
            config: RAW格式配置字典，默认使用900x1440, 10-bit, 3x3配置
        """
        if config is None:
            # 默认配置：对应5M-V1-P3-43格式
            self.config = {
                "row": 900,
                "col": 1440,
                "type": "uint16",
                "bit": 10,
                "byteOffset": 0,
                "cycleSize": [3, 3],
                "pages": 1,
                "dimOrder": "C",
                "shapeOrder": "PHW",
                "size": 2592000
            }
        else:
            self.config = config

        self.height = self.config["row"]
        self.width = self.config["col"]
        self.bit_depth = self.config["bit"]
        self.max_value = 2 ** self.bit_depth - 1  # 10-bit: 1023
        self.cycle_h, self.cycle_w = self.config["cycleSize"]
        self.num_channels = self.cycle_h * self.cycle_w

    def read(self, raw_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取RAW文件（完全模拟MATLAB的fread行为）

        参数:
            raw_path: RAW文件路径

        返回:
            channels: shape为(480, 300, 9)的多光谱图像
            raw_data: shape为(1440, 900)的原始数据
        """
        # 验证文件大小
        file_size = Path(raw_path).stat().st_size
        expected_size = self.config["size"]

        if file_size != expected_size:
            print(f"警告: 文件大小不匹配！")
            print(f"  期望: {expected_size} 字节")
            print(f"  实际: {file_size} 字节")

        # 读取二进制数据
        with open(raw_path, 'rb') as fid:
            # 跳过字节偏移
            if self.config["byteOffset"] > 0:
                fid.seek(self.config["byteOffset"])

            # 读取数据
            raw_data = np.fromfile(fid, dtype=np.uint16, count=self.height * self.width)

            # 完全模拟MATLAB的fread(fid, [1440 900], 'uint16')
            # MATLAB是列优先读取，所以需要用order='F'
            raw_data = raw_data.reshape((self.width, self.height), order='F')
            # 现在raw_data的shape是(1440, 900)，和MATLAB的rawData完全一致

        # 分解为多个通道（完全模拟MATLAB的循环）
        channels = self._demosaic_matlab_style(raw_data)

        return channels, raw_data

    def _demosaic_matlab_style(self, raw_data: np.ndarray) -> np.ndarray:
        """
        完全模拟MATLAB的通道分解逻辑

        MATLAB代码:
        for a=1:3
            for b=1:3
                m=1;
                for i=a:3:1440
                    n=1;
                    for j=b:3:900
                        img(m,n,k) = rawData(i,j);
                        n=n+1;
                    end
                    m=m+1;
                end
                k=k+1;
            end
        end

        参数:
            raw_data: shape为(1440, 900)的原始数据

        返回:
            channels: shape为(480, 300, 9)的多光谱图像
        """
        # 计算输出尺寸
        # MATLAB的i从a到1440，步长3，所以有480个值
        # MATLAB的j从b到900，步长3，所以有300个值
        m_size = len(range(0, self.width, self.cycle_h))  # 1440 // 3 = 480
        n_size = len(range(0, self.height, self.cycle_w))  # 900 // 3 = 300

        # 初始化通道数组
        channels = np.zeros((m_size, n_size, self.num_channels), dtype=np.uint16)

        k = 0
        # 完全模拟MATLAB的循环顺序
        for a in range(1, 4):  # MATLAB的1:3 对应 Python的1,2,3
            for b in range(1, 4):  # MATLAB的1:3 对应 Python的1,2,3
                m = 0  # Python从0开始
                for i in range(a, self.width + 1, 3):  # MATLAB的a:3:1440
                    n = 0  # Python从0开始
                    for j in range(b, self.height + 1, 3):  # MATLAB的b:3:900
                        # MATLAB索引从1开始，Python从0开始，所以需要-1
                        channels[m, n, k] = raw_data[i - 1, j - 1]
                        n += 1
                    m += 1
                k += 1

        return channels

    def normalize(self, data: np.ndarray, method: str = 'bit') -> np.ndarray:
        """
        归一化数据到[0, 1]（模拟MATLAB的 ./1024）

        参数:
            data: 输入数据
            method: 'bit'使用位深度归一化(除以1024), 'max'使用实际最大值归一化

        返回:
            归一化后的数据
        """
        if method == 'bit':
            # 模拟MATLAB的 ./1024
            return data.astype(np.float64) / (2 ** self.bit_depth)
        elif method == 'max':
            # 使用实际最大值归一化
            return data.astype(np.float64) / data.max()
        else:
            raise ValueError(f"未知的归一化方法: {method}")

    def save_channels(self, channels: np.ndarray, output_dir: str = 'output',
                      normalize: bool = True, format: str = 'bmp') -> None:
        """
        保存各个通道为图像文件（完全模拟MATLAB的imwrite）

        MATLAB代码:
        for i = 1:9
            filename= "img"+num2str(i)+".bmp";
            imwrite(img(:,:,i)./1024,filename);
        end

        参数:
            channels: shape为(480, 300, 9)的多光谱图像
            output_dir: 输出目录
            normalize: 是否归一化
            format: 输出格式 ('bmp', 'png', 'tiff')
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        for i in range(channels.shape[2]):
            channel = channels[:, :, i]

            if normalize:
                # 模拟MATLAB的 ./1024
                channel_norm = self.normalize(channel, method='bit')
                channel_norm = np.clip(channel_norm, 0, 1)

                # 转为uint8（OpenCV的imwrite会自动处理[0,1]的float）
                # 但为了完全一致，我们手动转换
                channel_8bit = (channel_norm * 255).astype(np.uint8)
            else:
                # 直接缩放到uint8范围
                channel_8bit = (channel * 255 // self.max_value).astype(np.uint8)

            # MATLAB的img是(480, 300, 9)，保存时不需要转置
            # 但OpenCV的imwrite期望的是(height, width)
            # MATLAB的imwrite会把(480, 300)保存为300x480的图片（宽x高）
            #
            channel_to_save = channel_8bit  # (480, 300) -> (300, 480)

            # 保存（模拟MATLAB的文件名格式）
            filename = output_path / f"channel{i + 1}.{format}"
            cv2.imwrite(str(filename), channel_to_save)
            print(f"已保存: {filename} (形状: {channel_to_save.shape})")

    def visualize_raw(self, raw_data: np.ndarray, title: str = "RAW Data") -> None:
        """
        可视化原始RAW数据

        参数:
            raw_data: shape为(1440, 900)的原始数据
            title: 窗口标题
        """
        # 归一化显示
        display = self.normalize(raw_data, method='max')
        display = (display * 255).astype(np.uint8)

        # MATLAB的imshow会自动调整显示方向，这里我们转置一下
        display = display.T

        cv2.imshow(title, display)
        print(f"按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_info(self) -> str:
        """获取配置信息"""
        info = f"""
多光谱RAW读取器配置:
  原始数据尺寸: {self.width} × {self.height} (MATLAB格式)
  数据类型: {self.config['type']}
  位深度: {self.bit_depth}-bit (最大值: {self.max_value})
  Bayer模式: {self.cycle_h} × {self.cycle_w}
  通道数: {self.num_channels}
  输出通道尺寸: {self.width // self.cycle_h} × {self.height // self.cycle_w}
  保存图片尺寸: {self.height // self.cycle_w} × {self.width // self.cycle_h} (宽×高)
  文件大小: {self.config['size']:,} 字节
        """
        return info.strip()


# ==================== 主程序 ====================

def main():
    """主函数示例"""

    # 1. 创建读取器
    reader = MultispectralRAWReader()

    # 2. 显示配置信息
    print(reader.get_info())
    print("\n" + "=" * 50 + "\n")

    # 3. 读取RAW文件
    raw_file = r"D:\Yee\Desktop\MSI\分区色温检测数据文件\人像场景\1\5M-V1-P3-43\2025-11-14-19-06-40\0-exp-3527us-gain-1-max-1023(1).raw"

    print(f"正在读取: {raw_file}")
    try:
        channels, raw_data = reader.read(raw_file)

        print(f"\n读取成功！")
        print(f"注意：MatLab默认行列是反的")
        print(f"  原始数据形状: {raw_data.shape}")
        print(f"  通道数据形状: {channels.shape}")
        print(f"  数据范围: [{raw_data.min()}, {raw_data.max()}]")

        # 4. 保存各个通道
        print(f"\n正在保存通道...")
        reader.save_channels(channels, output_dir='output_python', normalize=True)

        print(f"\n完成！")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {raw_file}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
