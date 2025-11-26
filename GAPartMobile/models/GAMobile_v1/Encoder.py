import torch
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy 
import open3d as o3d
from transformers import CLIPVisionModel
import numpy as np
import torchvision.transforms as T
import os

import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
sys.path.append(os.path.join(project_root, "third_party/Pointnet_Pointnet2_pytorch"))

# sys.path.append("/raid/wenbo/project/mshab/third_party/Pointnet_Pointnet2_pytorch") 
from models.pointnet2_cls_ssg import (
    get_model as Pointnet2,
)
from models.pointnet_cls import (
    get_model as Pointnet,
)

class MapEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, local_map):
        return self.cnn(local_map).flatten(2).permute(0, 2, 1)


class GoalEncoder(nn.Module):
    """抓取目标编码器"""
    def __init__(self, goal_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(goal_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
    def forward(self, goal_pose):
        return self.mlp(goal_pose).unsqueeze(1)
    
# class PointCloudEncoder(Pointnet2):
#     def __init__(self, embed_dim=256):
#         super(PointCloudEncoder, self).__init__(num_class=40, embed_dim=embed_dim, normal_channel=False)
#         self.feature_dim = 256

#     def forward(self, xyz):
#         if len(xyz.shape) == 2:
#             xyz = xyz.unsqueeze(0)
#         B, N, C = xyz.shape
#         xyz = xyz[:, :, :3]
#         xyz = xyz.permute(0, 2, 1)
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         return x.unsqueeze(1)

class PointCloudEncoder(Pointnet):
    def __init__(self, embed_dim=256):
        super(PointCloudEncoder, self).__init__(embed_dim=embed_dim, normal_channel=False)
        self.feature_dim = 256

    def forward(self, xyz):
        if len(xyz.shape) == 2:
            xyz = xyz.unsqueeze(0)
        xyz = xyz[:, :, :3]
        xyz = xyz.permute(0, 2, 1)
        x, _, _ = self.feat(xyz)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        return x.unsqueeze(1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ImageEncoder(nn.Module):
    """
    修改后的编码器，可以处理来自N个摄像头的图像输入（以字典形式提供），并自动适应数据格式。
    """
    def __init__(self, sample_obs, embed_dim=1024, encoder="clip"):
        super().__init__()
        self.encoder_type = encoder

        pixel_obs: Dict[str, torch.Tensor] = sample_obs["pixels"]
        state_obs: torch.Tensor = sample_obs["state"]
        output_feature = embed_dim
        state_feature = 1024
        extractor_out_features = 0

        if encoder == "clip":
            clip_model_name="openai/clip-vit-base-patch16"
            self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True)
            clip_output_dim = self.clip_vision_model.config.hidden_size
            clip_input_size = self.clip_vision_model.config.image_size

            # 使用 torchvision 进行图像尺寸调整
            self.clip_transform = T.Resize((clip_input_size, clip_input_size), antialias=True)
            # 冻结CLIP
            for param in self.clip_vision_model.parameters():
                param.requires_grad = False

            extractor_out_features = len(pixel_obs.keys()) * clip_output_dim
            # self.state_embed = nn.Sequential(
            #     nn.Linear(state_obs.size(-1), 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 128)
            # )
            # state_embed_dim = 128
            
            # projection_input_dim = clip_output_dim * 2 + state_embed_dim
            # self.projection = nn.Linear(projection_input_dim, embed_dim)
        
        elif encoder == "cnn":
            feature_size = 1024
            extractors = dict()

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
                with torch.no_grad():
                    n_flatten = cnn(pobs.float().cpu()).shape[1]
                    fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[k] = nn.Sequential(cnn, fc)
                extractor_out_features += feature_size
            # for state data we simply pass it through a single linear layer
            # extractors["state"] = nn.Linear(state_obs.size(-1), feature_size)
            # extractor_out_features += feature_size
            self.extractors = nn.ModuleDict(extractors)
            
        else:
            raise ValueError(f"Unsupported encoder type: '{encoder}'. Must be 'clip' or 'cnn'.")
        
        self.state_embed = nn.Linear(state_obs.size(-1), state_feature)
        extractor_out_features += state_feature
        if extractor_out_features == output_feature:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(extractor_out_features, output_feature)

    def forward(self, observations) -> torch.Tensor:
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]

        encoded_pixel_features = []

        if self.encoder_type == "clip":
            for key in sorted(pixels.keys()):
                pobs = pixels[key]
                # pobs = pobs.permute(0, 3, 1, 2)
                if pobs.shape[1] == 1:
                    pobs = pobs.repeat(1, 3, 1, 1) 
                pobs_transformed = self.clip_transform(pobs.float())
                vision_outputs = self.clip_vision_model(pixel_values=pobs_transformed)
                encoded_pixel_features.append(vision_outputs.pooler_output)
            image_features = torch.cat(encoded_pixel_features, dim=1)

        elif self.encoder_type == "cnn":
            
            for key, extractor in self.extractors.items():
                with torch.no_grad():
                    pobs = pixels[key].float()
                    # pobs = 1 / (1 + pixels[key].float() / 400)
                    pobs = 1 - torch.tanh(pixels[key].float() / 1000)
                    if len(pobs.shape) == 5:
                        b, fs, d, h, w = pobs.shape
                        pobs = pobs.reshape(b, fs * d, h, w)
                encoded_pixel_features.append(extractor(pobs))
            image_features = torch.cat(encoded_pixel_features, dim=1)
        
        state_features = self.state_embed(state)
        return self.projection(torch.cat([image_features, state_features], dim=1))



if __name__ == '__main__':
    print("--- Running a test of the Agent model ---")

    # 1. 定义测试用的超参数
    # 使用大于1的批次大小来确保模型能正确处理批处理数据
    BATCH_SIZE = 4
    STATE_FEATURES = 10
    ACTION_SHAPE = (2,)  # 假设是一个有2个动作的连续控制任务
    IMG_HEIGHT = 84
    IMG_WIDTH = 84

    # 2. 创建一个模拟的观测数据字典 (sample_obs)
    # 这个结构必须严格匹配模型在 __init__ 和 forward 方法中期望的结构
    sample_obs = {
        "pixels": {
            # 一个标准的4D图像观测 (Batch, Channels, Height, Width)
            # "camera_rgb": torch.randn(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH),
            # 一个5D图像观测, 例如包含帧堆叠 (Batch, FrameStack, Depth, Height, Width)
            # 这里的 Depth 和前面的 Channels 概念类似
            "camera_fs": torch.randn(BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH),
            "camera_fs_2": torch.randn(BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH)
        },
        "state": torch.randn(BATCH_SIZE, STATE_FEATURES)
    }

    # 3. 实例化 Agent 模型
    # 将模拟数据和动作形状传入构造函数
    
    model = ImageEncoder(sample_obs=sample_obs, encoder="clip")
    print("\nModel instantiated successfully.")
    # 如果需要，可以取消下面的注释来打印模型结构
    # print(model)
 
    print("\n--- Performing a forward pass ---")
    # forward 方法接收与构造函数中 sample_obs 相同结构的观测数据
    output_actions = model(sample_obs)
    print("Forward pass successful.")

    # 5. 检查输出结果
    print("\n--- Output ---")
    print(f"Output tensor shape: {output_actions.shape}")
    expected_shape = (BATCH_SIZE, np.prod(ACTION_SHAPE))
    print(f"Expected shape: {expected_shape}")
        
    print("\n--- Test finished ---")












    