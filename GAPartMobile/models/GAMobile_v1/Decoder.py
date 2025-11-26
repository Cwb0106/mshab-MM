# 在 Encoder.py 文件中新增以下类
import torch
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy 
import open3d as o3d
from transformers import CLIPVisionModel
import numpy as np
import torchvision.transforms as T


class PointDecoder(nn.Module):
    """
    一个轻量化的 MLP 解码器。
    用于从一个潜在特征向量，预测出点云编码器的特征向量。
    """
    def __init__(self, latent_dim, feature_dim, hidden_dim=512):
        """
        Args:
            latent_dim (int): 输入的潜在特征维度 (例如 latent_base_feature 的维度, 1024)
            feature_dim (int): 目标特征维度 (例如 PointNet Encoder 输出的维度, 256)
            hidden_dim (int): MLP 中间隐藏层的维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, latent_feature):
        return self.net(latent_feature)