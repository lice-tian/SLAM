# perform 3D reconstruction using PnP

import numpy as np
import cv2
import open3d as o3d

def solve_pnp_reconstruction(points3D, points2D):
    K = np.loadtxt('camera_intrinsic.txt')
    # 使用PnP算法求解相机的位姿
    # _, R_exp, t_exp = cv2.solvePnP(np.ascontiguousarray(points3D), points2D, K, np.zeros((5, 1)), flags=cv2.SOLVEPNP_ITERATIVE)

    _, R_exp, t_exp = cv2.solvePnP(np.ascontiguousarray(points3D), points2D, K,None)

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(R_exp)

    # 创建4x4的齐次变换矩阵
    camera_pose = np.hstack((R, t_exp))
    camera_pose = np.vstack((camera_pose, np.array([0, 0, 0, 1])))

    points3D_for_projection = points3D.reshape(-1, 1, 3)
    # 使用求解到的位姿反向投影2D点到3D空间
    # 注意：rvec 和 tvec 应该是从 solvePnP 返回的旋转和平移向量
    image_points, _ = cv2.projectPoints(points3D_for_projection, R_exp, t_exp, K, np.zeros((8, 1)))

    # 将返回的image_points转换回(n, 3)形状
    points3d_recovered = image_points.reshape(-1, 3).T

    return camera_pose, points3d_recovered

