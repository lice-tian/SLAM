# perform complete 3D reconstruction from 11 images
import os
import glob
import cv2
import numpy as np
from feature_extraction import extract_features
from feature_matching import match_features
from initial_recon import *
from pnp_recon import *
from bundle_adjustment import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def read_images(image_folder):
    image_files = glob.glob(os.path.join(images_folder, '*.png'))

    images = []
    for file_path in image_files:
        # 读入灰度图像
        image = cv2.imread(file_path, cv2.IMREAD_COLOR).astype(np.float32)
        images.append(image)
    images = np.array(images, dtype=np.float32)

    return images


if __name__ == '__main__':
    images_folder = '_images'
    images = read_images(images_folder)


    # 初始化特征点和匹配点集合
    features = []
    matches = []

    # SIFT特征提取
    features = extract_features(images)

    # 特征匹配
    matches = match_features(features)

    # 场景初始化
    image_index = 5
    first_image_matches = matches[image_index]
    first_keypoints = features[image_index][0]
    second_keypoints = features[image_index + 1][0]

    points3D,points2D = initialize_scene(first_keypoints, second_keypoints, first_image_matches)

    colors = get_point_colors(images[image_index], points2D)/255.0

    print(points3D[0:5])

    # temp = cv2.drawMatches(images[0].astype(np.uint8), first_keypoints, images[1].astype(np.uint8), second_keypoints, None)
    # temp = cv2.resize(temp, (1280, 720))
    # cv2.imshow('temp', temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    camera_pose, points_3d_recovered = solve_pnp_reconstruction(points3D, points2D)

    # 将恢复的3D点转换为open3d的PointCloud对象并可视化
    pcd = o3d.geometry.PointCloud()
    # points_3d_recovered = points3D
    pcd.points = o3d.utility.Vector3dVector(points_3d_recovered)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置点云的颜色

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到可视化对象
    vis.add_geometry(pcd)

    # 设置背景为黑色
    vis.get_render_option().background_color = [0, 0, 0]
    vis.get_render_option().point_size = 10.0  # 设置点的大小

    # 运行
    vis.run()

    # 销毁可视化窗口
    vis.destroy()


    # # 可视化三维点
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制三维点
    # ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])

    # # 设置坐标轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # # 显示图像
    # plt.show()
