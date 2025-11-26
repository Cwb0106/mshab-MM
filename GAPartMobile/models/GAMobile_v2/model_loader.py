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

# --- 子模块导入 ---
# (请确保你的路径设置正确)
try:
    # --- <<< 修改路径以匹配你的项目结构 >>> ---
    sys.path.append("/home/ubuntu/cwb_works/project/mshab/GAPartMobile/models/GAMobile_v2") 
    from Encoder import MapEncoder, GoalEncoder, PointCloudEncoder, ImageEncoder
    from Decoder import PointDecoder
    from ActionHead import FlowMatchingHead, ActionHead
    # from utils import ... (模型内部不需要 utils)
except ImportError as e:
    print("="*50)
    print(f"错误：无法导入子模块。请确保路径正确。错误: {e}")
    print("="*50)
    raise


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    
    """
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.generator = nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, feature, condition):
        """
        Use FiLM to integrate latent base action features into PC features.
        Args:
            feature (torch.Tensor): Features of the current frame's point cloud.
                                    Format: [batch_size, feature_dim], (e.g., [128, 1024])
                                    (e.g., [Batch_Size, Num_Points, feature_dim])
            condition (torch.Tensor): Latent base action features.
                                      Format: [batch_size, cond_dim], (e.g., [128, 1024])

        Returns:
            torch.Tensor: The baseaction-aware point cloud features.
                          Format: [batch_Size, output_feature_dim] (Same shape as input `feature`) (e.g., [128, 1024])
        """
        gamma_beta = self.generator(condition)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        output = gamma * feature + beta
        return output

class CrossAttentionLayer(nn.Module):
    """
    Cross Attenttion
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value):
        attn_output, _ = self.mha(query, key, value)
        output = self.norm(query + attn_output)
        return output

# =================================================================================
# 1. 长期记忆库 (PCMB)
# =================================================================================

class PerceptualCognitiveMemoryBank(nn.Module):
    """
    Perceptual-Cognitive Memory Bank (PCMB).

    This module acts as the core memory unit for MemoryVLA. It implements a read-write 
    memory mechanism that:
    1. Retrieves relevant historical context from a long-term memory bank based on the 
       current 'working memory' (current observation or preliminary intent).
    2. Fuses the retrieved history with the current context via a learned gating mechanism.
    3. Consolidates the information to update the memory bank slots for the next timestep.
    """
    
    def __init__(self, embed_dim, num_heads, memory_slots):
        super().__init__()
        self.memory_slots = memory_slots # Number of memory slots (e.g., K=16)
        self.embed_dim = embed_dim
        
        # 1. Retrieval Mechanism
        self.retrieval_attention = CrossAttentionLayer(embed_dim, num_heads)
        
        # 2. Fusion Mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # 3. Consolidation Mechanism
        # Learnable queries to reconstruct/update the memory bank
        self.memory_queries = nn.Parameter(torch.randn(1, self.memory_slots, embed_dim))
        self.consolidation_attention = CrossAttentionLayer(embed_dim, num_heads)

    def forward(self, working_memory_tokens, long_term_memory):
        """
        Memory Bank.

        Args:
            working_memory_tokens (torch.Tensor): Current time-step features acting as the query 
                                                  (e.g., preliminary base action intent).
                                                  Format: [Batch_Size, Seq_Len, embed_dim]
                                                  (e.g., [128, 1, 1024])
            long_term_memory (torch.Tensor): The persistent memory state from the previous step.
                                             Format: [Batch_Size, memory_slots, embed_dim]
                                             (e.g., [128, 16, 1024])

        Returns:
            tuple:
                - fused_policy_tokens (torch.Tensor): The memory-enhanced features.
                                                      Format: [Batch_Size, Seq_Len, embed_dim] (e.g., [128, 1, 1024])
                - new_long_term_memory (torch.Tensor): The updated long-term memory state for the next step.
                                                       Format: [Batch_Size, memory_slots, embed_dim] (e.g., [128, 16, 1024])
        """
        B = working_memory_tokens.shape[0]

        # --- 1. Retrieval ---
        # Query: Working Memory | Key/Value: Long Term Memory
        # The model looks into long_term_memory to find relevant past information.
        retrieved_memory = self.retrieval_attention(
            query=working_memory_tokens, 
            key=long_term_memory, 
            value=long_term_memory
        )

        # --- 2. Fusion ---
        # Gated fusion of current intent (working_memory) and retrieved history (retrieved_memory)
        gate_inputs = torch.cat([working_memory_tokens, retrieved_memory], dim=-1)
        gate = self.fusion_gate(gate_inputs)
        
        # Residual connection weighted by the gate
        fused_policy_tokens = (1 - gate) * working_memory_tokens + gate * retrieved_memory
        fused_policy_tokens = self.fusion_norm(fused_policy_tokens)
        
        # --- 3. Consolidation ---
        # Update long_term_memory. Concatenate the old long_term_memory with the newly fused tokens to form the candidate pool.
        # Fixed learned queries (memory_queries) extract information from this pool to form the new long_term_memory.
        candidate_memory = torch.cat([long_term_memory, fused_policy_tokens], dim=1)
        memory_queries = self.memory_queries.repeat(B, 1, 1)
        new_long_term_memory = self.consolidation_attention(
            query=memory_queries,
            key=candidate_memory,
            value=candidate_memory
        )
        return fused_policy_tokens, new_long_term_memory



class HierarchicalCrossAttentionNetwork(nn.Module):
    """
    Hierarchical Cross-Attention Network (HCAN).
    
    This architecture integrates a two-stage querying mechanism with a memory bank to handle 
    mobile manipulation tasks. It consists of two main branches:
    1. Base Branch: Generates navigation intent and queries the memory bank.
    2. End-Effector (EE) Branch: Uses 'imagined' point cloud features conditioned on mobility features.
    """
    def __init__(self, sample_obs, embed_dim=1024, num_heads=8,
                 goal_dim=7, num_points=1024, state_dim=16,
                 base_action_dim=2, ee_action_dim=7,
                 memory_slots=16): 
        super().__init__()
        self.embed_dim = embed_dim

        # --- 1. Encoders
        self.image_encoder = ImageEncoder(sample_obs, embed_dim=embed_dim, encoder="cnn")
        self.map_encoder = MapEncoder(embed_dim)
        self.goal_encoder = GoalEncoder(goal_dim, embed_dim)
        self.point_cloud_encoder = PointCloudEncoder(embed_dim)
        
        # 
        self.base_fusion = CrossAttentionLayer(embed_dim, num_heads)
        
        self.base_memory_bank = PerceptualCognitiveMemoryBank(
            embed_dim=embed_dim,
            num_heads=num_heads,
            memory_slots=memory_slots
        )

        # base action head
        self.base_action_head = ActionHead(in_features=embed_dim, action_dim=base_action_dim, head_type="mlp")

        # --- 3. End-Effector (EE) Branch
        self.film_fusion = FiLMLayer(cond_dim=embed_dim, feature_dim=embed_dim)
        self.ee_fusion = CrossAttentionLayer(embed_dim, num_heads)
        # EE action head
        self.ee_action_head = ActionHead(in_features=embed_dim, action_dim=ee_action_dim, head_type="mlp")

        # --- 4. Auxiliary Task
        self.point_decoder = PointDecoder(latent_dim=embed_dim, feature_dim=embed_dim)


    def forward(self, obs, prev_base_ltm): 
        
        # --- Step 1: Encode all modalities
        c_context = self.image_encoder(obs).unsqueeze(1) # [B, 1, D]
        f_map = self.map_encoder(obs['local_map'])       # [B, N_map, D]
        f_pc = self.point_cloud_encoder(obs['point_cloud']) # [B, 1, D]

        # --- Step 2: Base Branch -> f_mobility
        # Prepare navigation-specific features  (local occ map, obj_pose_wrt_base)
        if "obj_pose_wrt_base" in obs.keys():
            f_goal = self.goal_encoder(obs['obj_pose_wrt_base']) # [B, 1, D]
            nav_specialist_features = torch.cat([f_map, f_goal], dim=1)
        else:
            nav_specialist_features = f_map
            
        #   CrossAttention: Query=ImageContext, Key/Val=NavFeatures. Output: latent base action feature
        f_nav_intent_seq = self.base_fusion(
            query=c_context, 
            key=nav_specialist_features, 
            value=nav_specialist_features
        ) # Shape: [B, 1, D] (e.g., [128, 1, 1024])
        
        
        # c. Use 'f_nav_intent_seq' (as Working Memory) to query the Memory Bank
        #    Input Working Memory Shape: [B, 1, D]
        #    Input Long Term Memory Shape: [B, memory_slots, D]
        fused_memory_tokens, new_base_ltm = self.base_memory_bank(
            f_nav_intent_seq, 
            prev_base_ltm
        ) # fused_memory_tokens Shape: [B, 1, D] (e.g., [128, 1, 1024]); new_base_ltm Shape: [B, memory_slots, D] (e.g., [128, 16, 1024])
        f_mobility = fused_memory_tokens.squeeze(1) # Shape: [B, D] (e.g., [128, 1024])

        # e. Predict base action
        predicted_base_action = self.base_action_head(f_mobility)

        # --- Step 3: End-Effector Branch 
        # a. Using FiLM to integrate base action intent into pc features
        imagined_pc_feature = self.film_fusion(
            feature=f_pc.squeeze(1), 
            condition=f_mobility # memory-enhanced f_mobility
        ) # [B, D] (e.g., [128, 1024])
        imagined_pc_feature_seq = imagined_pc_feature.unsqueeze(1) # [B, 1, D] (e.g., [128, 1, 1024])

        #    CrossAttention: Query=memory-enhanced f_mobility, Key/Val=Imagined PC Features
        fused_ee_feature_seq = self.ee_fusion(
            query=f_mobility.unsqueeze(1), 
            key=imagined_pc_feature_seq, 
            value=imagined_pc_feature_seq
        ) # [B, 1, D] (e.g., [128, 1, 1024])
        
        # --- Step 4: Predict End-Effector Action
        fused_ee_feature = fused_ee_feature_seq.squeeze(1) # [B, D]
        # Residual connection
        ee_condition = fused_ee_feature + f_mobility 
        predicted_ee_action = self.ee_action_head(ee_condition)
        
        action_out = torch.cat([predicted_ee_action, predicted_base_action], dim=1)

        # --- Step 5: Auxiliary Task
        if self.training and obs.get('point_cloud_transformed_gt') is not None:
            with torch.no_grad():
                target_feature = self.point_cloud_encoder(obs['point_cloud_transformed_gt'])
            
            predicted_feature = imagined_pc_feature.unsqueeze(1) 

            # Return: action, new base memory, and auxiliary task data
            return action_out, new_base_ltm, (predicted_feature, target_feature)
        
        # During inference, return only final action and new chassis memory
        return action_out, new_base_ltm

   


if __name__ == '__main__':
    # 定义模型超参数
    BATCH_SIZE = 4
    EMBED_DIM = 1024
    IMG_SIZE = 128
    NUM_POINTS = 1024
    BASE_ACTION_DIM = 2
    EE_ACTION_DIM = 11
    GOAL_DIM = 7
    STATE_DIM = 42
    MEMORY_SLOTS = 16 

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模拟的输入数据字典
    pixels = {
        "fetch_head_depth": torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device),
        "fetch_hand_depth": torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE).to(device)
    }
    obs_data = {
        'pixels': pixels,
        'state': torch.randn(BATCH_SIZE, STATE_DIM).to(device),
        'local_map': torch.randn(BATCH_SIZE, 1, 75, 75).to(device),
        'obj_pose_wrt_base': torch.randn(BATCH_SIZE, GOAL_DIM).to(device),
        'point_cloud': torch.randn(BATCH_SIZE, NUM_POINTS, 3).to(device),
        'point_cloud_transformed_gt': torch.randn(BATCH_SIZE, NUM_POINTS, 3).to(device),
    }

    # 实例化模型 (V4.1)
    print("正在实例化模型 (V4.1 - 两阶段查询)...")
    model = HierarchicalCrossAttentionNetwork(
        sample_obs=obs_data,
        embed_dim=EMBED_DIM,
        goal_dim=GOAL_DIM,
        num_points=NUM_POINTS,
        state_dim=STATE_DIM,
        base_action_dim=BASE_ACTION_DIM,
        ee_action_dim=EE_ACTION_DIM,
        memory_slots=MEMORY_SLOTS 
    ).to(device)
    print("模型实例化成功。")

    
    # 初始化底盘记忆
    prev_base_ltm = torch.zeros(BATCH_SIZE, MEMORY_SLOTS, EMBED_DIM).to(device)
    print(f"初始化底盘长期记忆 (Base LTM) Shape: {prev_base_ltm.shape}")


    # =================================================================================
    # 2. 模块化性能分析 (与 V4 相同)
    # =================================================================================
    
    # --- 测试 1: 训练模式 (验证辅助任务分支) ---
    model.train() 
    print("\n--- 训练模式 (单步) 输出 ---")
    
    output_tuple_train = model(obs_data, prev_base_ltm)
    action_out_train, new_base_ltm_train, aux_data = output_tuple_train
    predicted_feature, target_feature = aux_data
    
    print(f"Action (train) Shape: {action_out_train.shape}")
    print(f"New Base LTM (train) Shape: {new_base_ltm_train.shape}")
    print(f"Predicted Feature Shape: {predicted_feature.shape}")
    print(f"Target Feature Shape: {target_feature.shape}")
    print("完成 (训练模式)")


    # --- 测试 2: 推理模式 (验证评估分支) ---
    model.eval()
    print("\n--- 推理模式 (单步) 输出 ---")
    
    with torch.no_grad():
        output_tuple_eval = model(obs_data, prev_base_ltm)
        action_out_eval, new_base_ltm_eval = output_tuple_eval
        
        print(f"Action (eval) Shape: {action_out_eval.shape}")
        print(f"New Base LTM (eval) Shape: {new_base_ltm_eval.shape}")
        print("完成 (推理模式)")