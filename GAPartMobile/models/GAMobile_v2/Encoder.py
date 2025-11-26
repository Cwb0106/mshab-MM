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
    """Grasp Goal Encoder"""
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
    Modified encoder that can handle image inputs from N cameras (provided as a dictionary) 
    and automatically adapts to data formats.
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

            # Use torchvision for image resizing
            self.clip_transform = T.Resize((clip_input_size, clip_input_size), antialias=True)
            # Freeze CLIP parameters
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

    # 1. Define hyperparameters for testing
    # Use a batch size greater than 1 to ensure the model handles batched data correctly
    BATCH_SIZE = 4
    STATE_FEATURES = 10
    ACTION_SHAPE = (2,)  # Assuming a continuous control task with 2 actions
    IMG_HEIGHT = 84
    IMG_WIDTH = 84

    # 2. Create a simulated observation dictionary (sample_obs)
    # This structure must strictly match the structure expected by the model in __init__ and forward methods
    sample_obs = {
        "pixels": {
            # A standard 4D image observation (Batch, Channels, Height, Width)
            # "camera_rgb": torch.randn(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH),
            # A 5D image observation, e.g., containing frame stacks (Batch, FrameStack, Depth, Height, Width)
            # Here 'Depth' is conceptually similar to Channels in this context
            "camera_fs": torch.randn(BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH),
            "camera_fs_2": torch.randn(BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH)
        },
        "state": torch.randn(BATCH_SIZE, STATE_FEATURES)
    }

    # 3. Instantiate the Agent model
    # Pass simulated data and action shape to the constructor
    
    model = ImageEncoder(sample_obs=sample_obs, encoder="cnn")
    print("\nModel instantiated successfully.")
    # Uncomment the following line to print the model structure if needed
    # print(model)
 
    print("\n--- Performing a forward pass ---")
    # The forward method receives observation data with the same structure as sample_obs in the constructor
    output_actions = model(sample_obs)
    print("Forward pass successful.")

    # 5. Check output results
    print("\n--- Output ---")
    print(f"Output tensor shape: {output_actions.shape}")
    expected_shape = (BATCH_SIZE, np.prod(ACTION_SHAPE))
    print(f"Expected shape: {expected_shape}")
        
    print("\n--- Test finished ---")