from collections import deque
from typing import Dict, List, Optional
import os
import glob
import json
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math
import ipdb
import os.path as osp

from scipy.spatial.transform import Rotation

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.common import flatten_state_dict
from mani_skill.utils import common

import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))
from GAPartMobile.utils import pose_7d_to_4x4_matrix, post_process_occupancy_grid
from GAPartMobile.utils import ROBOT_LINK_NAMES

DATASET_ASSETS_DIR = "/raid/wenbo/assets/maniskill/"
MANISKILL_DIR = "/raid/wenbo/project/mshab/ManiSkill/"

DATASET_CONFIG_DIR = osp.join(MANISKILL_DIR, "mani_skill/utils/scene_builder/replicacad/metadata/")
DATASET_ROOT = osp.join(DATASET_ASSETS_DIR, "data/scene_datasets/replica_cad_dataset/")

def pose_7d_to_4x4_matrix_torch(poses_7d: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion of 7D poses [x, y, z, w, x, y, z] to 4x4 transformation matrices.
    """
    num_envs = poses_7d.shape[0]
    positions = poses_7d[:, :3]
    quats_wxyz = poses_7d[:, 3:]
    w, x, y, z = quats_wxyz.unbind(dim=1)
    
    rot_matrices = torch.zeros(num_envs, 3, 3, device=poses_7d.device)
    rot_matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrices[:, 0, 1] = 2 * (x * y - w * z)
    rot_matrices[:, 0, 2] = 2 * (x * z + w * y)
    rot_matrices[:, 1, 0] = 2 * (x * y + w * z)
    rot_matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrices[:, 1, 2] = 2 * (y * z - w * x)
    rot_matrices[:, 2, 0] = 2 * (x * z - w * y)
    rot_matrices[:, 2, 1] = 2 * (y * z + w * x)
    rot_matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    transform_matrices = torch.zeros(num_envs, 4, 4, device=poses_7d.device)
    transform_matrices[:, :3, :3] = rot_matrices
    transform_matrices[:, :3, 3] = positions
    transform_matrices[:, 3, 3] = 1.0
    return transform_matrices


# ========================= Depth -> Point Cloud =========================

def _depth_to_pointcloud_torch_batched(depth_imgs, rgb_imgs, intrinsics, depth_trunc):
    """
    Converts batched depth images (N, H, W) to point clouds (N, H*W, 3).
    """
    N, H, W = depth_imgs.shape
    device = depth_imgs.device

    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    v, u = v.expand(N, H, W), u.expand(N, H, W)
    
    # Unit conversion: mm to meters
    z = depth_imgs * 0.001

    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cx, cy = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = torch.stack([x, y, z], dim=-1).view(N, H * W, 3)
    colors = (rgb_imgs.float() / 255.0).view(N, H * W, 3)
    
    # Filter based on depth range (e.g., > 10mm and < 2000mm)
    valid_mask = (depth_imgs > 10) & (depth_imgs < (depth_trunc * 1000.0))
    mask = valid_mask.view(N, H * W)
    
    return points, colors, mask


# ========================= Point Sampling (Random) =========================

def _random_sample_points_torch_batched(points, colors, mask, num_samples):
    """
    Batched random sampling of points from the valid mask.
    Currently uses random sampling instead of Farthest Point Sampling (FPS).
    """
    N, P, _ = points.shape
    device = points.device
    
    weights = mask.float()
    non_empty_mask = weights.sum(dim=1) > 0
    
    # Handle cases where no points are valid by assigning uniform weight to the first point
    if not non_empty_mask.all():
        weights[~non_empty_mask, 0] = 1
    
    indices = torch.multinomial(weights, num_samples, replacement=True)
    
    idx_expanded_xyz = indices.unsqueeze(-1).expand(-1, -1, 3)
    sampled_points = torch.gather(points, 1, idx_expanded_xyz)
    
    idx_expanded_rgb = indices.unsqueeze(-1).expand(-1, -1, 3)
    sampled_colors = torch.gather(colors, 1, idx_expanded_rgb)
    
    # Mask out samples that exceed the number of valid points
    num_valid_points = mask.sum(dim=1)
    valid_samples_mask = torch.arange(num_samples, device=device).expand(N, -1) < num_valid_points.unsqueeze(-1)
    
    sampled_points[~valid_samples_mask] = 0
    sampled_colors[~valid_samples_mask] = 0
    
    return sampled_points, sampled_colors


class OnlineValidationWrapper(gym.ObservationWrapper):
    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self._base_env: BaseEnv = env.unwrapped
        self.device = self._base_env.device
        self.all_map_data = {}
        self.max_map_points = 0

        print("Initializing OnlineValidationWrapper...")
        
        # 1. Load scene configurations (consistent with dataset.py)
        try:
            with open(os.path.join(DATASET_CONFIG_DIR, "scene_configs.json")) as f:
                build_config_json = json.load(f)
            self.build_configs = build_config_json.get("scenes", []) + build_config_json.get("staging_scenes", [])
            print(f"Loaded {len(self.build_configs)} scene configurations.")
        except Exception as e:
            print(f"FATAL: Could not read scene_configs.json: {e}")
            self.build_configs = []

        # 2. Pre-load all scene maps
        print("Pre-loading all scene maps...")
 
        for scene_id in self.build_configs:
            # 3. Construct .obj file path
            map_file_path = os.path.join(DATASET_ROOT, "configs", "scenes", scene_id.split(".json")[0] + ".fetch.navigable_positions.obj")

            if not os.path.exists(map_file_path):
                continue

            try:
                with open(map_file_path, 'r') as f:
                    coords = [ (float(p[1]), float(p[2])) for l in f if l.strip().startswith('v ') and len((p := l.strip().split())) >= 3 ]
                
                if coords:
                    x_coords, y_coords = zip(*coords)
                    self.all_map_data[scene_id] = (np.array(x_coords), np.array(y_coords))
                    self.max_map_points = max(self.max_map_points, len(x_coords))
                else:
                    self.all_map_data[scene_id] = (np.array([]), np.array([]))
            except Exception as e:
                print(f"Warning: Failed to load map {scene_id} from {map_file_path}: {e}")

        print(f"Successfully pre-loaded {len(self.all_map_data)} maps. Max points in a map: {self.max_map_points}")

    # ========================= Local Map Generation =========================

    def _get_local_map_vectorized(self, robot_poses_7d: torch.Tensor, map_ids_np: np.array):
        """
        Generates local occupancy maps centered on the robot for a batch of environments.
        """
        num_envs = robot_poses_7d.shape[0]
        scene_names = [self.build_configs[mid] for mid in map_ids_np]

        batched_map_points_np = np.zeros((num_envs, 2, self.max_map_points), dtype=np.float32)
        for i, scene_id in enumerate(scene_names):
            x, y = self.all_map_data.get(scene_id, (np.array([]), np.array([])))
            if x.size > 0:
                num_pts = len(x)
                batched_map_points_np[i, 0, :num_pts] = x
                batched_map_points_np[i, 1, :num_pts] = y
        
        batched_map_points = torch.from_numpy(batched_map_points_np).to(self.device)

        robot_xy = robot_poses_7d[:, :2]
        quats_wxyz = robot_poses_7d[:, 3:]
        robot_yaws = torch.atan2(2 * (quats_wxyz[:,0]*quats_wxyz[:,3] + quats_wxyz[:,1]*quats_wxyz[:,2]), 1 - 2 * (quats_wxyz[:,2]**2 + quats_wxyz[:,3]**2))
        
        map_points_translated = batched_map_points - robot_xy[:, :, None]
        c, s = torch.cos(robot_yaws), torch.sin(robot_yaws)
        rot_matrices = torch.stack([torch.stack([s, -c], dim=1), torch.stack([c, s], dim=1)], dim=1)
        map_points_rotated = torch.bmm(rot_matrices, map_points_translated)
        
        resolution, map_size_m = 0.02, 1.5
        grid_size = int(map_size_m / resolution)
        indices = ((map_points_rotated + map_size_m / 2) / resolution).long()
        
        mask = (indices[:, 0, :] >= 0) & (indices[:, 0, :] < grid_size) & (indices[:, 1, :] >= 0) & (indices[:, 1, :] < grid_size)
        valid_point_mask = batched_map_points.abs().sum(dim=1) > 0
        mask &= valid_point_mask
        
        batch_idx = torch.arange(num_envs, device=self.device)[:, None].expand(-1, self.max_map_points)
        valid_batch_idx = batch_idx[mask]
        valid_indices = indices.permute(0, 2, 1)[mask]
        
        occupancy_grids = torch.zeros((num_envs, grid_size, grid_size), dtype=torch.uint8, device=self.device)
        occupancy_grids[valid_batch_idx, valid_indices[:, 1], valid_indices[:, 0]] = 255
        
        grids_for_dilate = occupancy_grids.unsqueeze(1).float()
        dilated_grids = F.max_pool2d(grids_for_dilate, kernel_size=5, stride=1, padding=2)
        return dilated_grids.squeeze(1).byte()

    def map_loader(self, map_info_file: str):
        if map_info_file in self.map_cache:
            return self.map_cache[map_info_file]

        scene_id = map_info_file.split(".json")[0]
        map_obj_path = os.path.join(DATASET_ROOT, "configs", "scenes", f"{scene_id}.fetch.navigable_positions.obj")
        
        x_coords, y_coords = [], []
        try:
            with open(map_obj_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('v '):
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            x_coords.append(float(parts[1]))
                            y_coords.append(float(parts[2]))
        except Exception as e:
            print(f"Warning: Failed to load or parse map file {map_obj_path}: {e}")

        self.map_cache[map_info_file] = (np.array(x_coords), np.array(y_coords))
        return self.map_cache[map_info_file]

    # ========================= Point Cloud Merging =========================

    def _get_merged_pc_vectorized(self, sensor_data, sensor_params, base_poses_7d):
        num_envs = base_poses_7d.shape[0]
        base_poses_4x4 = pose_7d_to_4x4_matrix_torch(base_poses_7d)
        T_w_to_b = torch.inverse(base_poses_4x4)
        T_cv_to_gl = torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], device=self.device, dtype=torch.float32)

        # Process Head Camera
        head_points, head_colors, head_mask = _depth_to_pointcloud_torch_batched(
            sensor_data["fetch_head"]["depth"], sensor_data["fetch_head"]["rgb"],
            sensor_params["fetch_head"]["intrinsic_cv"], depth_trunc=2.0
        )
        head_extrinsics = T_w_to_b @ (sensor_params["fetch_head"]["cam2world_gl"] @ T_cv_to_gl)
        points_homo = torch.cat([head_points, torch.ones(num_envs, head_points.shape[1], 1, device=self.device)], dim=2)
        transformed_head_points = torch.bmm(points_homo, head_extrinsics.transpose(1, 2))[:, :, :3]

        # Process Hand Camera
        hand_points, hand_colors, hand_mask = _depth_to_pointcloud_torch_batched(
            sensor_data["fetch_hand"]["depth"], sensor_data["fetch_hand"]["rgb"],
            sensor_params["fetch_hand"]["intrinsic_cv"], depth_trunc=2.0
        )
        hand_extrinsics = T_w_to_b @ (sensor_params["fetch_hand"]["cam2world_gl"] @ T_cv_to_gl)
        points_homo = torch.cat([hand_points, torch.ones(num_envs, hand_points.shape[1], 1, device=self.device)], dim=2)
        transformed_hand_points = torch.bmm(points_homo, hand_extrinsics.transpose(1, 2))[:, :, :3]
        
        # Spatial Filtering
        final_head_mask = head_mask & \
                          (transformed_head_points[:, :, 2] >= 0.1) & \
                          (transformed_head_points[:, :, 2] < 1.7)
        
        final_hand_mask = hand_mask & \
                          (transformed_hand_points[:, :, 2] >= 0.1) & \
                          (transformed_hand_points[:, :, 2] < 1.7) & \
                          (transformed_hand_points[:, :, 0] >= 0)

        # Sampling (Biased towards hand points)
        num_total, hand_ratio = 1024, 0.6
        num_hand, num_head = int(num_total * hand_ratio), num_total - int(num_total * hand_ratio)

        sampled_head_points, sampled_head_colors = _random_sample_points_torch_batched(
            transformed_head_points, head_colors, final_head_mask, num_head
        )
        sampled_hand_points, sampled_hand_colors = _random_sample_points_torch_batched(
            transformed_hand_points, hand_colors, final_hand_mask, num_hand
        )
        
        final_points = torch.cat([sampled_head_points, sampled_hand_points], dim=1)
        final_colors = torch.cat([sampled_head_colors, sampled_hand_colors], dim=1)
        
        return torch.cat([final_points, final_colors], dim=2)

    # ========================= Observation Processing =========================

    # NOTE: Conventional experiments are currently commented out in favor of pi0 eval.
    def observation(self, observation: Dict) -> Dict:
        if 'base_pos_wrt_world' in observation['extra']:
            robot_poses_tensor = observation['extra']['base_pos_wrt_world']
        else:  
            robot_poses_tensor = vectorize_pose(self.base_env.agent.base_link.pose)
        
        map_ids_tensor = self._base_env.build_config_idxs
        local_maps_tensor = self._get_local_map_vectorized(robot_poses_tensor, map_ids_tensor)
    
        sensor_data = observation["sensor_data"]
        if sensor_data["fetch_head"]["depth"].ndim == 4:
            sensor_data["fetch_head"]["depth"] = sensor_data["fetch_head"]["depth"].squeeze(-1)
        if sensor_data["fetch_hand"]["depth"].ndim == 4:
            sensor_data["fetch_hand"]["depth"] = sensor_data["fetch_hand"]["depth"].squeeze(-1)
        
        point_clouds_tensor = self._get_merged_pc_vectorized(
            sensor_data, observation["sensor_param"], robot_poses_tensor
        )
        
        agent_obs_t = observation["agent"]
        extra_obs_t = observation["extra"]
    
        if 'is_grasped' in extra_obs_t and extra_obs_t['is_grasped'].ndim == 1:
            extra_obs_t['is_grasped'] = extra_obs_t['is_grasped'].unsqueeze(1)
            
        extra_obs_for_state = {k: v for k, v in extra_obs_t.items() if k != 'base_pos_wrt_world'}
    
        # Conventional State Construction
        # del extra_obs_for_state['obj_pose_wrt_base']
        # del extra_obs_for_state['goal_pos_wrt_base']
        agent_state_list = list(agent_obs_t.values())
        extra_state_list = list(extra_obs_for_state.values())
        robot_state_tensor = torch.cat(agent_state_list + extra_state_list, dim=1)
    
        final_obs = {
            'state': robot_state_tensor,   # B, 42
            'local_map': local_maps_tensor.unsqueeze(1),      # B, 1, 75, 75
            'point_cloud': point_clouds_tensor[:, :, :3],    # B, 1024, 3
            # 'obj_pose_wrt_base': extra_obs_for_state['obj_pose_wrt_base'],
        }
    
        final_obs['pixels'] = dict(
            fetch_head_depth=observation["sensor_data"]["fetch_head"]["depth"].unsqueeze(1),
            fetch_hand_depth=observation["sensor_data"]["fetch_hand"]["depth"].unsqueeze(1),
        )
    
        return final_obs
