# perform 3D reconstruction using PnP

import numpy as np
import cv2
import open3d as o3d

def solve_pnp_reconstruction(points3D, points2D):
    K = np.loadtxt('camera_intrinsic.txt')
    #flags=SOLVEPNP_EPNP
    _, R_exp, t_exp = cv2.solvePnP(
        np.ascontiguousarray(points3D),
        points2D,
        K,
        np.zeros((5, 1)),
        flags=cv2.SOLVEPNP_EPNP
    )

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(R_exp)

    # 创建4x4的齐次变换矩阵
    camera_pose = np.vstack((np.hstack((R, t_exp)), np.array([0, 0, 0, 1])))

    # 为points3D添加一个1的列向量，使其成为齐次坐标 (n, 4)
    points3D_homogeneous = np.hstack((points3D, np.ones((points3D.shape[0], 1))))

    # 使用相机位姿变换三维点到相机坐标系
    points3D_transformed = camera_pose.dot(points3D_homogeneous.T).T

    # 截取前三个值作为变换后的三维点坐标
    points3D_transformed = points3D_transformed[:, :3]

    return camera_pose, points3D_transformed

