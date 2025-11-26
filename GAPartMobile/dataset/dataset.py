from mshab.utils.dataset import ClosableDataLoader, ClosableDataset
from mshab.utils.array import to_tensor
import open3d as o3d
import math
import torch
import torch.utils.data # <-- 【新增】为了使用 default_collate
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import h5py
import os
import os.path as osp
import json
from omegaconf import OmegaConf
import numpy as np
import glob
import ipdb

import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))
from GAPartMobile.models.GAMobile_v2.utils import base_action_to_transform_matrix, transform_point_cloud
from GAPartMobile.utils import pose_7d_to_4x4_matrix, post_process_occupancy_grid
from GAPartMobile.utils import ROBOT_LINK_NAMES

DATASET_ASSETS_DIR = "/raid/wenbo/assets/maniskill/"
MANISKILL_DIR = "/raid/wenbo/project/mshab/ManiSkill/"
DATASET_CONFIG_DIR = MANISKILL_DIR + "mani_skill/utils/scene_builder/replicacad/metadata/"

include_staging_scenes = True


class GABCDataset(ClosableDataset):
    def __init__(
        self,
        data_dir_fp: str,
        max_cache_size: int,
        transform_fn=torch.from_numpy,
        trajs_per_obj: Union[str, int] = "all",
        cat_state=False,
        cat_pixels=False,
        sequence_length: int = 6 
    ):
        data_dir_fp: Path = Path(data_dir_fp)
        self.data_files: List[h5py.File] = []
        self.json_files: List[Dict] = []
        self.obj_names_in_loaded_order: List[str] = []
        self.map_cache = {} 
        
        self.K = sequence_length 

        if data_dir_fp.is_file():
            data_file_names = [data_dir_fp.name]
            data_dir_fp = data_dir_fp.parent
        else:
            data_file_names = os.listdir(data_dir_fp)
     
        for data_fn in data_file_names:
            if data_fn.endswith(".h5"):
                json_fn = data_fn.replace(".h5", ".json")
                self.data_files.append(h5py.File(data_dir_fp / data_fn, "r"))
                with open(data_dir_fp / json_fn, "rb") as f:
                    self.json_files.append(json.load(f))
                self.obj_names_in_loaded_order.append(data_fn.replace(".h5", ""))

        self.dataset_idx_to_data_idx = dict()
        dataset_idx = 0
        
        for file_idx, json_file in enumerate(self.json_files):
            # Sample trajectories
            if trajs_per_obj == "all":
                use_ep_jsons = json_file["episodes"]
            else:
                use_ep_jsons = random.sample(json_file["episodes"], k=trajs_per_obj)

            for ep_json in use_ep_jsons:
                ep_id = ep_json["episode_id"]
                map_id = ep_json['build_config_idxs']

                # Only index valid start steps for a sequence of length K
                # The last valid start index is (total_steps - K).
                # range(valid_steps) produces 0 ... (total - K), which is correct.
                valid_steps = ep_json["elapsed_steps"] - self.K + 1

                for step in range(valid_steps): 
                    # 'step' here represents the START of a sequence
                    self.dataset_idx_to_data_idx[dataset_idx] = (file_idx, ep_id, step, map_id)
                    dataset_idx += 1
        
        self._data_len = dataset_idx
        self.max_cache_size = max_cache_size
        self.cache = dict()
        self.transform_fn = transform_fn
        self.cat_state = cat_state
        self.cat_pixels = cat_pixels

        with open(osp.join(project_root, "GAPartMobile/config/robot_id.json"), 'r') as f:
                robot_link_id_map = json.load(f)
        
        # Load robot link IDs dynamically
        self.robot_id = set(robot_link_id_map.values())

        with open(osp.join(DATASET_CONFIG_DIR, "scene_configs.json")) as f:
            build_config_json = json.load(f)
            self.build_configs = build_config_json["scenes"]
            if include_staging_scenes:
                self.build_configs += build_config_json["staging_scenes"]

    def transform_idx(self, x, data_index):
        if isinstance(x, h5py.Group) or isinstance(x, dict):
            return dict((k, self.transform_idx(v, data_index)) for k, v in x.items())
        out = self.transform_fn(np.array(x[data_index]))
        if len(out.shape) == 0:
            out = out.unsqueeze(0)
        return out

    def map_loader(self, file_path):
        # Check cache first
        if file_path in self.map_cache:
            return self.map_cache[file_path] 

        x_coords = []
        y_coords = []
        
        obj_path = DATASET_ASSETS_DIR + "/data/scene_datasets/replica_cad_dataset/configs/scenes/" + file_path.split(".json")[0]+".fetch.navigable_positions.obj"
        with open(obj_path, 'r') as f:
            for line in f:
                if line.strip().startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            x_coords.append(x)
                            y_coords.append(y)
                        except ValueError:
                            pass

        self.map_cache[file_path] = (x_coords, y_coords)
        return x_coords, y_coords

    def _get_local_map(self, map_x_coords, map_y_coords, robot_pose_7d):
        """
        Generates a local occupancy grid map centered on the robot.
        """
        # 1. Parse robot pose
        robot_x = robot_pose_7d[0]
        robot_y = robot_pose_7d[1]
        qw, qx, qy, qz = robot_pose_7d[3:]
        robot_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # 2. Transform map points to robot frame
        map_points = np.array([map_x_coords, map_y_coords])
        map_points_translated = map_points - np.array([[robot_x], [robot_y]])
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        rotation_matrix = np.array([[s, -c], [c, s]])
        map_points_rotated = rotation_matrix @ map_points_translated

        # 3. Create 2D occupancy grid
        resolution = 0.02
        map_size_m = 1.5
        grid_size = int(map_size_m / resolution)
        occupancy_grid_sparse = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        local_map_mask = (np.abs(map_points_rotated[0, :]) <= map_size_m / 2) & \
                        (np.abs(map_points_rotated[1, :]) <= map_size_m / 2)
        local_map_points = map_points_rotated[:, local_map_mask]

        if local_map_points.shape[1] > 0:
            grid_x_indices = ((local_map_points[0, :] + map_size_m / 2) / resolution).astype(int)
            grid_y_indices = ((local_map_points[1, :] + map_size_m / 2) / resolution).astype(int)
            grid_x_indices = np.clip(grid_x_indices, 0, grid_size - 1)
            grid_y_indices = np.clip(grid_y_indices, 0, grid_size - 1)
            occupancy_grid_sparse[grid_y_indices, grid_x_indices] = 255

        # 4. Post-process (dilate)
        occupancy_grid_processed = post_process_occupancy_grid(
            occupancy_grid_sparse, kernel_size=4, iterations=1
        )
        return occupancy_grid_processed

    def _get_merged_pc(
        self,
        head_rgb: torch.Tensor,
        head_depth: torch.Tensor,
        head_seg: torch.Tensor, 
        head_intrinsic: torch.Tensor,
        head_cam2world: torch.Tensor,
        hand_rgb: torch.Tensor,
        hand_depth: torch.Tensor,
        hand_seg: torch.Tensor, 
        hand_intrinsic: torch.Tensor,
        hand_cam2world: torch.Tensor,
        base_pose_7d: np.ndarray,
    ):
        """
        Merges point clouds from head and hand cameras, applies spatial filtering, 
        and performs biased random sampling (preferring hand points).
        Returns point cloud (xyz+rgb) and segmentation IDs.
        """
        # 1. Calculate transformation matrices
        base_pos_wrt_world_4x4 = pose_7d_to_4x4_matrix(base_pose_7d)
        T_world_to_base = np.linalg.inv(base_pos_wrt_world_4x4)
        T_cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

        # 2. Process Head Point Cloud
        T_head_cam_gl_to_world = head_cam2world.numpy()
        extrinsic_head_to_base = T_world_to_base @ (T_head_cam_gl_to_world @ T_cv_to_gl)
        head_points, head_colors, head_seg_ids = self.create_colored_pointcloud_from_tensors(
            rgb_tensor=head_rgb, depth_tensor=head_depth, seg_tensor=head_seg,
            intrinsic_tensor=head_intrinsic, extrinsic_tensor=torch.from_numpy(extrinsic_head_to_base), depth_trunc=2.0
        )

        # 3. Process Hand Point Cloud
        T_hand_cam_gl_to_world = hand_cam2world.numpy()
        extrinsic_hand_to_base = T_world_to_base @ (T_hand_cam_gl_to_world @ T_cv_to_gl)
        hand_points, hand_colors, hand_seg_ids = self.create_colored_pointcloud_from_tensors(
            rgb_tensor=hand_rgb, depth_tensor=hand_depth, seg_tensor=hand_seg,
            intrinsic_tensor=hand_intrinsic, extrinsic_tensor=torch.from_numpy(extrinsic_hand_to_base), depth_trunc=2.0
        )

        # 4. Filter point clouds based on workspace limits
        head_mask = (head_points[:, 2] >= 0.1) & (head_points[:, 2] < 1.7)
        filtered_head_points = head_points[head_mask]
        filtered_head_colors = head_colors[head_mask]
        filtered_head_seg_ids = head_seg_ids[head_mask]

        hand_mask = (hand_points[:, 2] >= 0.1) & (hand_points[:, 2] < 1.7) & (hand_points[:, 0] >= 0)
        filtered_hand_points = hand_points[hand_mask]
        filtered_hand_colors = hand_colors[hand_mask]
        filtered_hand_seg_ids = hand_seg_ids[hand_mask]
        
        # 5. Biased Random Sampling (Sample more from hand than head)
        num_total, hand_ratio = 1024, 0.6
        num_hand, num_head = int(num_total * hand_ratio), num_total - int(num_total * hand_ratio)
        
        # Sample Head
        num_available_head = len(filtered_head_points)
        if num_available_head > 0:
            head_indices = np.random.choice(num_available_head, num_head, replace=True)
            sampled_head_points = filtered_head_points[head_indices]
            sampled_head_colors = filtered_head_colors[head_indices]
            sampled_head_seg_ids = filtered_head_seg_ids[head_indices]
        else:
            sampled_head_points = np.zeros((num_head, 3), dtype=np.float32)
            sampled_head_colors = np.zeros((num_head, 3), dtype=np.float32)
            sampled_head_seg_ids = np.zeros(num_head, dtype=np.uint32)

        # Sample Hand
        num_available_hand = len(filtered_hand_points)
        if num_available_hand > 0:
            hand_indices = np.random.choice(num_available_hand, num_hand, replace=True)
            sampled_hand_points = filtered_hand_points[hand_indices]
            sampled_hand_colors = filtered_hand_colors[hand_indices]
            sampled_hand_seg_ids = filtered_hand_seg_ids[hand_indices]
        else:
            sampled_hand_points = np.zeros((num_hand, 3), dtype=np.float32)
            sampled_hand_colors = np.zeros((num_hand, 3), dtype=np.float32)
            sampled_hand_seg_ids = np.zeros(num_hand, dtype=np.uint32)

        # 6. Merge and Return
        final_points = np.vstack([sampled_head_points, sampled_hand_points])
        final_colors = np.vstack([sampled_head_colors, sampled_hand_colors])
        final_seg_ids = np.concatenate([sampled_head_seg_ids, sampled_hand_seg_ids])

        point_cloud_with_color = np.hstack([final_points, final_colors])
        return point_cloud_with_color, final_seg_ids

    def get_single_item(self, index):
        """
        Retrieves a temporal sequence of observations and actions.
        Returns: (obs_seq, act_seq, episode_starts_seq)
        """
        # Caching sequences is expensive, so it is disabled.
        
        # 1. Get sequence start info
        file_num, ep_num, start_step, map_id = self.dataset_idx_to_data_idx[index]
        
        obs_list = []
        act_list = []
        episode_start_list = []

        # 2. Build the sequence by iterating K times
        for t in range(self.K):
            step_num = start_step + t
            
            # 3. Retrieve single frame data using helper
            obs, act = self._get_frame_data(file_num, ep_num, step_num, map_id)
            
            obs_list.append(obs)
            act_list.append(act)
            
            # 4. Mark the start of the sequence (t=0) for RNN/Transformer reset
            episode_start_list.append(t == 0)

        # 5. Collate lists into a batch (Sequence dimension becomes 0)
        # obs_seq dict values will be [K, ...]
        obs_seq = torch.utils.data.default_collate(obs_list)
        act_seq = torch.stack(act_list, dim=0) # [K, ActionDim]
        episode_starts_seq = torch.tensor(episode_start_list, dtype=torch.bool) # [K]

        res = (obs_seq, act_seq, episode_starts_seq)
        return res

    def _get_frame_data(self, file_num, ep_num, step_num, map_id):
        """
        Helper function to load and process data for a single time step.
        """
        ep_data = self.data_files[file_num][f"traj_{ep_num}"]
        ep_json = next((ep for ep in self.json_files[file_num]["episodes"] if ep["episode_id"] == ep_num), None)

        # --- Load Data and Process Point Clouds ---
        observation = ep_data["obs"]
        agent_obs = self.transform_idx(observation["agent"], step_num)
        extra_obs = self.transform_idx(observation["extra"], step_num)
        extra_obs["base_pos_wrt_world"] = self.transform_idx(ep_data["base_pos_wrt_world"], step_num)
        
        map_info = self.build_configs[map_id]
        map_x, map_y = self.map_loader(map_info)
        occupancy_grid_processed = self._get_local_map(map_x, map_y, extra_obs["base_pos_wrt_world"].numpy())
        
        # Load sensors
        fetch_head_depth = self.transform_idx(observation["sensor_data"]["fetch_head"]["depth"], step_num)
        fetch_hand_depth = self.transform_idx(observation["sensor_data"]["fetch_hand"]["depth"], step_num)
        fetch_head_rgb = self.transform_idx(observation["sensor_data"]["fetch_head"]["rgb"], step_num)
        fetch_hand_rgb = self.transform_idx(observation["sensor_data"]["fetch_hand"]["rgb"], step_num)
        fetch_head_seg = self.transform_idx(observation["sensor_data"]["fetch_head"]["segmentation"], step_num)
        fetch_hand_seg = self.transform_idx(observation["sensor_data"]["fetch_hand"]["segmentation"], step_num)
        fetch_head_intrinsic = self.transform_idx(observation["sensor_param"]["fetch_head"]["intrinsic_cv"], step_num)
        fetch_hand_intrinsic = self.transform_idx(observation["sensor_param"]["fetch_hand"]["intrinsic_cv"], step_num)
        fetch_head_cam2world = self.transform_idx(observation["sensor_param"]["fetch_head"]["cam2world_gl"], step_num)
        fetch_hand_cam2world = self.transform_idx(observation["sensor_param"]["fetch_hand"]["cam2world_gl"], step_num)
        
        # Merge Point Cloud
        point_cloud_with_color, final_seg_ids = self._get_merged_pc(
            head_rgb=fetch_head_rgb, head_depth=fetch_head_depth, head_seg=fetch_head_seg,
            head_intrinsic=fetch_head_intrinsic, head_cam2world=fetch_head_cam2world,
            hand_rgb=fetch_hand_rgb, hand_depth=fetch_hand_depth, hand_seg=fetch_hand_seg,
            hand_intrinsic=fetch_hand_intrinsic, hand_cam2world=fetch_hand_cam2world,
            base_pose_7d=extra_obs["base_pos_wrt_world"].numpy(),
        )

        is_robot_mask = np.isin(final_seg_ids, list(self.robot_id))

        # --- Compute Ground Truth Transformation (Step t -> t+1) ---
        act = self.transform_idx(ep_data["actions"], step_num)

        if step_num >= ep_json["elapsed_steps"] - 1:
            # End of trajectory, no future motion
            point_cloud_transformed_gt = point_cloud_with_color[:, :3]
        else:
            # a. Get robot pose at t and t+1
            pose_t_7d = extra_obs["base_pos_wrt_world"].numpy()
            pose_t1_7d = self.transform_idx(ep_data["base_pos_wrt_world"], step_num + 1).numpy()

            # b. Convert to 4x4 matrices
            T_world_base_t = pose_7d_to_4x4_matrix(pose_t_7d)
            T_world_base_t1 = pose_7d_to_4x4_matrix(pose_t1_7d)

            # c. Compute relative transform: T_{t+1 <- t}
            transform_matrix = np.linalg.inv(T_world_base_t1) @ T_world_base_t

            # d. Transform current cloud to match future frame
            point_cloud_transformed_gt = self.transform_point_cloud_np(
                point_cloud_with_color[:, :3],
                transform_matrix
            )
            # Robot points stay attached to base, reset them
            point_cloud_transformed_gt[is_robot_mask] = point_cloud_with_color[:, :3][is_robot_mask]

        # --- Final Observation Dict ---
        del extra_obs['base_pos_wrt_world']

        obs = dict()
        obs['state'] = torch.cat([*agent_obs.values(),*extra_obs.values()])
        obs['local_map'] = torch.from_numpy(occupancy_grid_processed[np.newaxis, :, :])
        obs['point_cloud'] = torch.from_numpy(point_cloud_with_color[:, :3])
        obs['pixels'] = dict(
            fetch_head_depth=fetch_head_depth.permute(2, 0, 1),
            fetch_hand_depth=fetch_hand_depth.permute(2, 0, 1),
        )
        obs['point_cloud_transformed_gt'] = torch.from_numpy(point_cloud_transformed_gt)
        
        return obs, act

    def transform_point_cloud_np(self, point_cloud: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Applies a 4x4 NumPy transformation matrix to a NumPy point cloud."""
        num_points = point_cloud.shape[0]
        if num_points == 0:
            return point_cloud
        pcd_homogeneous = np.hstack([point_cloud, np.ones((num_points, 1))])
        transformed_pcd_homogeneous = pcd_homogeneous @ transform_matrix.T
        return transformed_pcd_homogeneous[:, :3]

    def _depth_to_pointcloud(self, depth_img: np.ndarray, rgb_img: np.ndarray, seg_img: np.ndarray, intrinsic_matrix: np.ndarray, depth_trunc: float):
        """
        Back-projects depth image to 3D point cloud and extracts segmentation IDs.
        """
        height, width = depth_img.shape
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        v, u = np.mgrid[0:height, 0:width]
        
        depth_trunc_mm = depth_trunc * 1000
        valid_mask = (depth_img > 10) & (depth_img < depth_trunc_mm) 
        
        u_flat = u[valid_mask]
        v_flat = v[valid_mask]
        depth_flat = depth_img[valid_mask]
        
        z = depth_flat / 1000.0
        x = (u_flat - cx) * z / fx
        y = (v_flat - cy) * z / fy

        points = np.vstack((x, y, z)).T
        colors = rgb_img[valid_mask] / 255.0
        
        # Extract actor ID (first channel) if segmentation is 3D (H, W, 4) or 2D
        if seg_img.ndim == 3:
            seg_ids = seg_img[:, :, 0][valid_mask]
        else: 
            seg_ids = seg_img[valid_mask]
            
        return points, colors, seg_ids 


    def create_colored_pointcloud_from_tensors(
        self,
        rgb_tensor: torch.Tensor,
        depth_tensor: torch.Tensor,
        seg_tensor: torch.Tensor, 
        intrinsic_tensor: torch.Tensor,
        extrinsic_tensor: torch.Tensor,
        depth_trunc: float = 2.0,
    ):
        """
        Wrapper to convert tensors to numpy, perform back-projection, and transform points to base frame.
        """
        rgb_numpy = rgb_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
        depth_numpy = depth_tensor.squeeze(2).cpu().numpy()
        seg_numpy = seg_tensor.cpu().numpy() 
        intrinsic_numpy = intrinsic_tensor.cpu().numpy()
        extrinsic_numpy = extrinsic_tensor.cpu().numpy()

        points_camera_frame, colors, seg_ids = self._depth_to_pointcloud( 
            depth_img=depth_numpy,
            rgb_img=rgb_numpy,
            seg_img=seg_numpy, 
            intrinsic_matrix=intrinsic_numpy,
            depth_trunc=depth_trunc
        )
        
        num_points = points_camera_frame.shape[0]
        if num_points == 0:
            return np.array([]).reshape(0,3), np.array([]).reshape(0,3), np.array([]).reshape(0)

        points_homogeneous = np.hstack([points_camera_frame, np.ones((num_points, 1))])
        points_transformed_homogeneous = (extrinsic_numpy @ points_homogeneous.T).T
        points_base_frame = points_transformed_homogeneous[:, :3]
        
        return points_base_frame, colors, seg_ids 


    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            return self.get_single_item(indexes)
        return [self.get_single_item(i) for i in indexes]

    def __len__(self):
        return self._data_len

    def close(self):
        for f in self.data_files:
            f.close()


if __name__ == "__main__":
    data_dir_fp = "/raid/wenbo/project/mshab/mshab_exps/gen_data_save_trajectories/set_table/pick/train/013_apple/"
    max_cache_size=0
    cat_state=False
    cat_pixels=False
    trajs_per_obj="all"
    
    sequence_length = 1 
    dataset = GABCDataset(
        data_dir_fp=data_dir_fp, 
        max_cache_size=max_cache_size, 
        cat_state=cat_state, 
        cat_pixels=cat_pixels, 
        trajs_per_obj=trajs_per_obj,
        sequence_length=sequence_length 
    )
    
    bc_dataloader = ClosableDataLoader(
        dataset,
        batch_size=64,
        shuffle=True, 
        num_workers=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("dataset: length", len(dataset))
    
    from tqdm import tqdm
    progress_bar = tqdm(
        bc_dataloader, 
        leave=False 
    )

    for obs_seq, act_seq, episode_starts_seq in iter(progress_bar):
        first_key = list(obs_seq.keys())[0]
        print(f"Batch Obs Shape (e.g., {first_key}): {obs_seq[first_key].shape}") 
        print(f"Batch Act Shape: {act_seq.shape}") 
        print(f"Batch Episode Starts Shape: {episode_starts_seq.shape}") 
        break