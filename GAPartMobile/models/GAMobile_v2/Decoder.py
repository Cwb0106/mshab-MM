# 在 Encoder.py 文件中新增以下类
import torch
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy 
import open3d as o3d
import numpy as np
import torchvision.transforms as T


class PointDecoder(nn.Module):
    def __init__(self, latent_dim, feature_dim, hidden_dim=512):
        """
        Args:
            latent_dim (int): Dimension of the input latent feature (e.g., dimension of latent_base_feature, 1024).
            feature_dim (int): Dimension of the target feature (e.g., output dimension of PointNet Encoder, 256).
            hidden_dim (int): Dimension of the hidden layers in the MLP.
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