# perform 3D reconstruction using PnP

import numpy as np
import cv2
import open3d as o3d


def get_point_colors(image, points2D):
    """
    根据点的坐标在彩色图像中获取颜色。

    参数:
    - image: OpenCV图像对象，必须为彩色图像。
    - points2D: 一个形状为 (n, 2) 的NumPy数组，包含点的二维坐标。

    返回:
    - colors: 一个形状为 (n, 3) 的NumPy数组，包含每个点的BGR颜色值。
    """
    colors = []
    for pt in points2D:
        x, y = int(pt[0]), int(pt[1])  # 转换为整数坐标
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # 获取BGR颜色值
            color = image[y, x, :3]
            colors.append(color)
        else:
            # 如果点的坐标超出图像范围，可以将其颜色设置为黑色或其他
            colors.append([0, 0, 0])
    colors = np.array(colors)
    return colors


def solve_pnp_reconstruction(points3D, points2D, K):
    #flags=SOLVEPNP_EPNP
    _, R_exp, t_exp = cv2.solvePnP(
        np.ascontiguousarray(points3D),
        points2D,
        K,
        np.zeros((5, 1)),
        flags=cv2.SOLVEPNP_EPNP
    )

    # # 将旋转向量转换为旋转矩阵
    # R, _ = cv2.Rodrigues(R_exp)

    # # 创建4x4的齐次变换矩阵
    # camera_pose = np.vstack((np.hstack((R, t_exp)), np.array([0, 0, 0, 1])))

    # # 为points3D添加一个1的列向量，使其成为齐次坐标 (n, 4)
    # points3D_homogeneous = np.hstack((points3D, np.ones((points3D.shape[0], 1))))

    # # 使用相机位姿变换三维点到相机坐标系
    # points3D_transformed = camera_pose.dot(points3D_homogeneous.T).T

    # # 截取前三个值作为变换后的三维点坐标
    # points3D_transformed = points3D_transformed[:, :3]

    # return camera_pose, points3D_transformed, R_exp, t_exp
    return R_exp, t_exp

