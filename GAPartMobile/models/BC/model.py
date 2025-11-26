from typing import Dict
import time # 引入 time 模块

from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, sample_obs, single_act_shape):
        super().__init__()

        extractors = dict()

        extractor_out_features = 0
        feature_size = 1024

        pixel_obs: Dict[str, torch.Tensor] = sample_obs["pixels"]
        state_obs: torch.Tensor = sample_obs["state"]

        # 为每个像素输入创建CNN特征提取器
        for k, pobs in pixel_obs.items():
            if len(pobs.shape) == 5:
                b, fs, d, h, w = pobs.shape
                pobs = pobs.reshape(b, fs * d, h, w)
            pobs_stack = pobs.size(1)
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=pobs_stack,
                    out_channels=24,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=24,
                    out_channels=36,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=36,
                    out_channels=48,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=48,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            # 动态计算CNN输出维度并连接一个FC层
            with torch.no_grad():
                n_flatten = cnn(pobs.float().cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            
            extractors[k] = nn.Sequential(cnn, fc)
            extractor_out_features += feature_size

        # 为状态输入创建一个简单的线性层
        extractors["state"] = nn.Linear(state_obs.size(-1), feature_size)
        extractor_out_features += feature_size

        self.extractors = nn.ModuleDict(extractors)
        
        # 最后的MLP层，用于融合所有特征
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(extractor_out_features, 2048)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, np.prod(single_act_shape)),
                std=0.01 * np.sqrt(2),
            ),
        )

    def forward(self, observations) -> torch.Tensor:
        # pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]
        encoded_tensor_list = []

        # 分别处理不同来源的观测数据
        for key, extractor in self.extractors.items():
            if key == "state":
                encoded_tensor_list.append(extractor(state))
            else:
                # 像素数据的预处理
                pobs = pixels[key].float()
                # pobs = 1 / (1 + pixels[key].float() / 400) # 原始代码中的其他预处理方式
                pobs = 1 - torch.tanh(pobs / 1000)
                if len(pobs.shape) == 5:
                    b, fs, d, h, w = pobs.shape
                    pobs = pobs.reshape(b, fs * d, h, w)
                encoded_tensor_list.append(extractor(pobs))
        
        # 拼接特征并通过MLP
        return self.mlp(torch.cat(encoded_tensor_list, dim=1))


# =================================================================================
# 性能分析脚本
# =================================================================================
if __name__ == '__main__':
    # 1. 定义模拟输入数据的超参数
    BATCH_SIZE = 4
    STATE_DIM = 28
    ACTION_DIM = 8
    NUM_ITERATIONS = 100 # 测试循环次数

    # 定义多个模拟的像素输入源 (例如，不同角度的摄像头)
    # 结构: BATCH_SIZE, FrameStack, Depth, Height, Width
    PIXEL_OBS_SHAPES = {
        "front_camera_rgb": (BATCH_SIZE, 1, 3, 96, 96),
        "hand_camera_depth": (BATCH_SIZE, 1, 1, 64, 64),
    }

    # 2. 创建用于模型初始化的样本观测数据
    sample_pixels = {k: torch.randn(*s) for k, s in PIXEL_OBS_SHAPES.items()}
    sample_state = torch.randn(BATCH_SIZE, STATE_DIM)
    sample_obs_for_init = {"pixels": sample_pixels, "state": sample_state}

    # 3. 设置设备并实例化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Agent(sample_obs_for_init, (ACTION_DIM,)).to(device)
    model.eval() # 切换到评估模式进行推理测试

    # 4. 创建用于性能分析的实际输入数据 (并移动到目标设备)
    observations = {
        "pixels": {k: v.to(device) for k, v in sample_pixels.items()},
        "state": sample_state.to(device)
    }
    
    # =================================================================================
    # 5. 模块化性能分析
    # =================================================================================
    print("\n" + "="*50)
    print(" " * 15 + "模块化性能分析")
    print("="*50)
    
    with torch.no_grad():
        # --- 5.1 逐个测试特征提取器 (Extractors) ---
        # 首先对像素数据进行预处理，避免在循环中重复计算
        processed_pixels = {}
        for key, pobs in observations['pixels'].items():
            pobs_processed = 1 - torch.tanh(pobs.float() / 1000)
            if len(pobs_processed.shape) == 5:
                b, fs, d, h, w = pobs_processed.shape
                pobs_processed = pobs_processed.reshape(b, fs * d, h, w)
            processed_pixels[key] = pobs_processed
            
        # 循环测试每个提取器
        for key, extractor in model.extractors.items():
            start_time = time.time()
            for _ in range(NUM_ITERATIONS):
                if key == "state":
                    _ = extractor(observations['state'])
                else:
                    _ = extractor(processed_pixels[key])
                if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            print(f"模块: Extractor '{key}',    耗时: {end_time - start_time:.4f} 秒 / {NUM_ITERATIONS} 次")
            
        # --- 5.2 测试最后的MLP层 ---
        # 需要先准备好MLP的输入
        encoded_tensors = []
        for key, extractor in model.extractors.items():
            if key == "state":
                encoded_tensors.append(extractor(observations['state']))
            else:
                encoded_tensors.append(extractor(processed_pixels[key]))
        mlp_input = torch.cat(encoded_tensors, dim=1)
        
        start_time = time.time()
        for _ in range(NUM_ITERATIONS):
            _ = model.mlp(mlp_input)
            if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        print(f"模块: Final MLP,                耗时: {end_time - start_time:.4f} 秒 / {NUM_ITERATIONS} 次")
        
    # =================================================================================
    # 6. 整体流程性能分析
    # =================================================================================
    print("\n" + "="*50)
    print(" " * 15 + "整体流程性能分析")
    print("="*50)
    
    with torch.no_grad():
        # --- 6.1 测试完整的 model.forward ---
        start_time = time.time()
        for _ in range(NUM_ITERATIONS):
            _ = model(observations)
            if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / NUM_ITERATIONS) * 1000
        print(f"流程: model.forward (完整推理), 耗时: {total_time:.4f} 秒 / {NUM_ITERATIONS} 次, 平均每次: {avg_time_ms:.2f} ms")