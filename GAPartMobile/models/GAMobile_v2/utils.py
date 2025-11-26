import torch
import numpy as np
import torch.nn.functional as F


def base_action_to_transform_matrix(base_action: torch.Tensor) -> torch.Tensor:
    """
    ### MODIFICATION START ###
    Converts a batch of 2D base actions [dx_forward, d_theta] into 4x4 rigid body transformation matrices.
    This version is simplified to handle only 2D actions.
    ### MODIFICATION END ###

    Args:
        base_action: Tensor of shape (B, 2) [dx_forward, d_theta]

    Returns:
        torch.Tensor: Transformation matrix of shape (B, 4, 4)
    """
    B = base_action.shape[0]
    device = base_action.device
    
    if base_action.shape[1] != 2:
        raise ValueError(f"This function now expects a 2D base_action, but got shape: {base_action.shape}")

    dx_forward, d_theta = base_action[:, 0], base_action[:, 1]
    
    cos_theta = torch.cos(d_theta)
    sin_theta = torch.sin(d_theta)
    
    # Construct transformation matrix
    transform_matrix = torch.zeros(B, 4, 4, device=device)
    transform_matrix[:, 0, 0] = cos_theta
    transform_matrix[:, 0, 1] = -sin_theta
    transform_matrix[:, 1, 0] = sin_theta
    transform_matrix[:, 1, 1] = cos_theta
    transform_matrix[:, 2, 2] = 1.0
    transform_matrix[:, 3, 3] = 1.0
    transform_matrix[:, 0, 3] = dx_forward # Translation occurs only along the robot's local x-axis
    
    return transform_matrix


def transform_point_cloud(point_cloud: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
    """
    Transforms the point cloud using the transformation matrix (this function remains unchanged).
    """
    B, N, _ = point_cloud.shape
    pcd_homogeneous = F.pad(point_cloud, (0, 1), mode='constant', value=1.0)
    transformed_pcd_homogeneous = torch.bmm(transform_matrix, pcd_homogeneous.transpose(1, 2))
    transformed_pcd = transformed_pcd_homogeneous[:, :3, :].transpose(1, 2)
    return transformed_pcd