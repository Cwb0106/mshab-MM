import torch
import numpy as np
import torch.nn.functional as F


def base_action_to_transform_matrix(base_action: torch.Tensor) -> torch.Tensor:
    """
    ### MODIFICATION START ###
    将一个批次的2维底盘动作 [dx_forward, d_theta] 转换为 4x4 的刚体变换矩阵。
    这个版本被简化，只处理2维动作。
    ### MODIFICATION END ###

    Args:
        base_action: shape (B, 2) 的张量 [dx_forward, d_theta]

    Returns:
        torch.Tensor: shape (B, 4, 4) 的变换矩阵
    """
    B = base_action.shape[0]
    device = base_action.device
    
    if base_action.shape[1] != 2:
        raise ValueError(f"This function now expects a 2D base_action, but got shape: {base_action.shape}")

    dx_forward, d_theta = base_action[:, 0], base_action[:, 1]
    
    cos_theta = torch.cos(d_theta)
    sin_theta = torch.sin(d_theta)
    
    # 构建变换矩阵
    transform_matrix = torch.zeros(B, 4, 4, device=device)
    transform_matrix[:, 0, 0] = cos_theta
    transform_matrix[:, 0, 1] = -sin_theta
    transform_matrix[:, 1, 0] = sin_theta
    transform_matrix[:, 1, 1] = cos_theta
    transform_matrix[:, 2, 2] = 1.0
    transform_matrix[:, 3, 3] = 1.0
    transform_matrix[:, 0, 3] = dx_forward # 平移只在机器人自己的x轴上
    
    return transform_matrix


def transform_point_cloud(point_cloud: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
    """
    使用变换矩阵来变换点云 (此函数保持不变)。
    """
    B, N, _ = point_cloud.shape
    pcd_homogeneous = F.pad(point_cloud, (0, 1), mode='constant', value=1.0)
    transformed_pcd_homogeneous = torch.bmm(transform_matrix, pcd_homogeneous.transpose(1, 2))
    transformed_pcd = transformed_pcd_homogeneous[:, :3, :].transpose(1, 2)
    return transformed_pcd

