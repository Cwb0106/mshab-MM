from scipy.spatial.transform import Rotation
import numpy as np
import cv2 



ROBOT_LINK_NAMES = [
    "root", "root_arm_1_link_1", "root_arm_1_link_2", "base_link", "r_wheel_link",
    "l_wheel_link", "torso_lift_link", "head_pan_link", "head_tilt_link",
    "head_camera_link", "head_camera_rgb_frame", "head_camera_rgb_optical_frame",
    "head_camera_depth_frame", "head_camera_depth_optical_frame", "shoulder_pan_link",
    "shoulder_lift_link", "upperarm_roll_link", "elbow_flex_link", "forearm_roll_link",
    "wrist_flex_link", "wrist_roll_link", "gripper_link", "r_gripper_finger_link",
    "l_gripper_finger_link", "bellows_link", "bellows_link2", "estop_link",
    "laser_link", "torso_fixed_link"
]

def pose_7d_to_4x4_matrix(pose_7d):
    """
    将一个7维的位姿向量 [pos(x,y,z), quat(w,x,y,z)] 转换为 4x4 的齐次变换矩阵。

    Args:
        pose_7d (np.ndarray): 形状为 (7,) 的位姿向量。

    Returns:
        np.ndarray: 形状为 (4, 4) 的齐次变换矩阵。
    """
    # 1. 从7维向量中分离位置和平移
    position = pose_7d[:3]
    quaternion_wxyz = pose_7d[3:]

    # 2. 将四元数转换为 3x3 旋转矩阵
    # 注意：scipy的from_quat方法需要 [x, y, z, w] 的顺序！
    # 所以我们需要重新排列我们的 [w, x, y, z]
    quaternion_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    
    rotation_matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()

    # 3. 将旋转矩阵和平移向量组合成 4x4 的齐次变换矩阵
    # 首先创建一个4x4的单位矩阵
    transform_matrix = np.eye(4)
    # 将3x3的旋转矩阵放入左上角
    transform_matrix[:3, :3] = rotation_matrix
    # 将3x1的位置向量放入右上角
    transform_matrix[:3, 3] = position
    
    return transform_matrix


def post_process_occupancy_grid(grid, kernel_size=5, iterations=3):
    # 删掉或注释掉第2步的调试代码

    if grid is None or grid.size == 0:
        return np.zeros_like(grid) if grid is not None else None

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 这是最关键的一行，请确保这里是 dilate
    processed_grid = cv2.dilate(grid, kernel, iterations=iterations)

    return processed_grid