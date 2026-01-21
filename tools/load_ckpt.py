import torch
import lightning as L
from omegaconf import OmegaConf
import os

# 定义输出文件路径
output_file = '/data2/ylc/codes/IndoorGau/tools/checkpoint_analysis.txt'

# 打开输出文件
with open(output_file, 'w') as f:
    # 加载checkpoint文件
    checkpoint_path = '/data2/ylc/codes/IndoorGau/outputs/ddp2_internet_test1/internet-e0.ckpt'

    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    f.write(f"Loading checkpoint from: {checkpoint_path}\n")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 打印checkpoint的键
    f.write("\nCheckpoint keys:\n")
    for key in checkpoint.keys():
        f.write(f"- {key}\n")

    # 加载模型状态字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        f.write("\nModel state dict keys:\n")
        
        # 按层次组织参数
        layers = {}
        for key, value in state_dict.items():
            # 分割键名，获取层次结构
            parts = key.split('.')
            layer_name = '.'.join(parts[:-1]) if len(parts) > 1 else parts[0]
            param_name = parts[-1]
            
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name][param_name] = value
        
        # 打印每一层的参数
        f.write("\nModel layers and parameters:\n")
        for layer_name, params in layers.items():
            f.write(f"\nLayer: {layer_name}\n")
            f.write(f"  Number of parameters: {len(params)}\n")
            for param_name, param_value in params.items():
                f.write(f"  - {param_name}: shape={param_value.shape}, dtype={param_value.dtype}\n")
    else:
        f.write("\nNo state_dict found in checkpoint.\n")

    f.write("\nCheckpoint loading completed.\n")

print(f"Checkpoint analysis completed. Results saved to: {output_file}")

