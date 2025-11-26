import torch
import torch.nn as nn
import torch.nn.functional as F
# 新增的import，用于加载CLIP模型
from transformers import CLIPVisionModel
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import ipdb
import os
import time # 引入 time 模块

import sys

sys.path.append("/raid/wenbo/project/mshab/GAPartMobile/models/GAMobile")
from Encoder import MapEncoder, GoalEncoder, PointCloudEncoder, ImageEncoder
from Decoder import PointDecoder
from ActionHead import FlowMatchingHead, ActionHead
from utils import base_action_to_transform_matrix, transform_point_cloud
# =================================================================================
# 0. 基础/辅助模块 (保持不变)
# =================================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) 层。
    使用一个条件向量来生成缩放(gamma)和偏置(beta)参数，
    并将其应用于另一个特征向量。
    """
    def __init__(self, cond_dim, feature_dim):
        """
        Args:
            cond_dim (int): 条件向量的维度 (例如 f_mobility 的维度)。
            feature_dim (int): 要被调制的特征向量的维度 (例如 f_pc 的维度)。
        """
        super().__init__()
        # 这个线性层将条件向量映射为 gamma 和 beta
        # 输出维度是 feature_dim 的两倍 (一个用于gamma, 一个用于beta)
        self.generator = nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, feature, condition):
        """
        Args:
            feature (torch.Tensor): 要被调制的特征 (f_pc)。
            condition (torch.Tensor): 条件向量 (f_mobility)。
        """
        # 1. 从条件向量生成 gamma 和 beta
        gamma_beta = self.generator(condition)
        
        # 2. 将输出切分为 gamma 和 beta
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        # 3. 应用 FiLM 公式: output = gamma * feature + beta
        output = gamma * feature + beta
        return output

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value):
        attn_output, _ = self.mha(query, key_value, key_value)
        output = self.norm(query + attn_output)
        return output


# =================================================================================
# 3. 最终的整体网络 (更新主干流部分)
# =================================================================================
class HierarchicalCrossAttentionNetwork(nn.Module):
    def __init__(self, sample_obs, embed_dim=1024, num_heads=8,
                 goal_dim=7, num_points=1024, state_dim=16,
                 base_action_dim=2, ee_action_dim=7, # 默认 base_action_dim=2
                 clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_encoder = ImageEncoder(sample_obs, embed_dim=embed_dim, encoder="cnn")
        self.map_encoder = MapEncoder(embed_dim)
        self.goal_encoder = GoalEncoder(goal_dim, embed_dim)
        self.point_cloud_encoder = PointCloudEncoder(embed_dim)
        self.point_decoder = PointDecoder(latent_dim=embed_dim, feature_dim=embed_dim)
        
        self.base_fusion = CrossAttentionLayer(embed_dim, num_heads)
        self.ee_fusion = CrossAttentionLayer(embed_dim, num_heads)
        self.base_action_head = ActionHead(in_features=embed_dim, action_dim=base_action_dim, head_type="mlp")
        self.ee_action_head = ActionHead(in_features=embed_dim, action_dim=ee_action_dim, head_type="mlp")

        self.film_fusion = FiLMLayer(cond_dim=embed_dim, feature_dim=embed_dim)

    def forward(self, obs): # 在forward中加入gt_base_action
            # --- 第1步: 编码通用特征和移动意图 (f_mobility) ---
        c_context = self.image_encoder(obs).unsqueeze(1)
        f_map = self.map_encoder(obs['local_map'])
        if "obj_pose_wrt_base" in obs.keys():
            f_goal = self.goal_encoder(obs['obj_pose_wrt_base'])
            nav_specialist_features = torch.cat([f_map, f_goal], dim=1)
        else:
            nav_specialist_features = f_map
            
        fused_base_feature_seq = self.base_fusion(query=c_context, key_value=nav_specialist_features)
        f_mobility = fused_base_feature_seq.squeeze(1) # [B, 1024]

        # --- 第2步: 预测底盘动作 (主任务) ---
        predicted_base_action = self.base_action_head(f_mobility)

        # --- 第3步: 实现您设想的 "想象力" 融合流程 ---

        # a. 获取当前的点云特征
        f_pc = self.point_cloud_encoder(obs['point_cloud']) # [B, 1, 1024]

        # b. 使用 FiLM 层进行融合，得到“想象中”的特征
        #    f_pc 需要先去掉序列维度，因为FiLM处理的是扁平向量
        imagined_pc_feature = self.film_fusion(feature=f_pc.squeeze(1), condition=f_mobility) # [B, 1024]
        
        # c. 将想象出的特征准备成 Key 和 Value 用于交叉注意力
        imagined_pc_feature_seq = imagined_pc_feature.unsqueeze(1) # [B, 1, 1024]

        # d. 使用 f_mobility 作为 Query，查询“想象中”的点云特征
        fused_ee_feature_seq = self.ee_fusion(query=f_mobility.unsqueeze(1), key_value=imagined_pc_feature_seq)
        
        # --- 第4步: 预测机械臂动作 (主任务) ---
        fused_ee_feature = fused_ee_feature_seq.squeeze(1)
        ee_condition = fused_ee_feature + f_mobility
        predicted_ee_action = self.ee_action_head(ee_condition)
        
        action_out = torch.cat([predicted_ee_action, predicted_base_action], dim=1)

        # --- <<< MODIFICATION: 修改辅助任务逻辑 >>> ---
        if self.training and obs['point_cloud_transformed_gt'] is not None:
            # 1. 获取变换后点云的“真实”目标特征
            with torch.no_grad(): # 确保 Encoder 在这里不接受梯度
                target_feature = self.point_cloud_encoder(obs['point_cloud_transformed_gt'])
            
            # 2. 我们“想象”出的特征就是 FiLM 层的输出
            predicted_feature = imagined_pc_feature.unsqueeze(1) # 增加序列维度以匹配target_feature

            return action_out, predicted_feature, target_feature
        
        # 在推理时，只返回最终动作
        return action_out

   


if __name__ == '__main__':
    # =================================================================================
    # 1. 初始化设置 (模型、数据、设备)
    # =================================================================================
    # 定义模型超参数
    BATCH_SIZE = 4
    RGB_CAMERA_NAMES = ['fetch_head_rgb', 'fetch_hand_rgb']
    depth_CAMERA_NAMES = ['fetch_head_depth', 'fetch_hand_depth']
    EMBED_DIM = 1024
    # 注意：标准的CLIP模型需要224x224的输入, 你的代码中已经通过interpolate处理了
    IMG_SIZE = 128
    NUM_POINTS = 1024
    BASE_ACTION_DIM = 2
    EE_ACTION_DIM = 11
    GOAL_DIM = 7
    STATE_DIM = 16
    NUM_ITERATIONS = 1 # 定义测试循环次数

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # 创建模拟的输入数据字典
    rgb_images_dict = {
        name: torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE, 3).to(device) for name in RGB_CAMERA_NAMES
    }
    depth_images_dict = {
        name: torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE).to(device) for name in depth_CAMERA_NAMES
    }


    batched_point_cloud = torch.randn(BATCH_SIZE, NUM_POINTS, 3).to(device)

    pixels = {
        "fetch_head_depth": torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device),
        "fetch_hand_depth": torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE).to(device)
    }

    obs_data = {

        'pixels': pixels,
        'state': torch.randn(BATCH_SIZE, STATE_DIM).to(device),
        'local_map': torch.randn(BATCH_SIZE, 1, 75, 75).to(device),
        # 'obj_pose_wrt_base': torch.randn(BATCH_SIZE, GOAL_DIM).to(device),
        'point_cloud': batched_point_cloud.to(device),
        'point_cloud_transformed_gt': batched_point_cloud.to(device),
        'noisy_base_action': torch.randn(BATCH_SIZE, BASE_ACTION_DIM).to(device),
        'noisy_ee_action': torch.randn(BATCH_SIZE, EE_ACTION_DIM).to(device),
        'time_step': torch.rand(BATCH_SIZE, 1).to(device)
    }

    # 实例化模型
    model = HierarchicalCrossAttentionNetwork(
        sample_obs=obs_data,
        embed_dim=EMBED_DIM,
        goal_dim=GOAL_DIM,
        num_points=NUM_POINTS,
        state_dim=STATE_DIM,
        base_action_dim=BASE_ACTION_DIM,
        ee_action_dim=EE_ACTION_DIM,
        clip_model_name="openai/clip-vit-base-patch16"
    ).to(device)
    # model.eval() # 使用 .eval() 模式进行推理测试
    # =================================================================================
    # 2. 模块化性能分析
    # =================================================================================
    with torch.no_grad(): 
        output = model(obs_data)
        print(output)
        print("完成")
   