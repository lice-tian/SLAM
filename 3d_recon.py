# perform complete 3D reconstruction from 11 images
import os
import glob
import cv2
import numpy as np
from feature_extraction import extract_features
from feature_matching import match_features
from initial_recon import *
from pnp_recon import *
from bundle_adjustment import optimize_scene


def read_images(image_folder):
    image_files = glob.glob(os.path.join(image_folder, '*.png'))

    images = []
    for file_path in image_files:
        # 读入灰度图像
        image = cv2.imread(file_path, cv2.IMREAD_COLOR).astype(np.float32)
        images.append(image)
    images = np.array(images, dtype=np.float32)

    return images


if __name__ == '__main__':
    # 相机内参矩阵
    K = np.loadtxt('camera_intrinsic.txt')
    
    images_folder = 'images'
    images = read_images(images_folder)
    
    # SIFT特征提取
    features = extract_features(images)

    base_image_index = 5
    base_image = images[base_image_index]
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    base_image_feature = features[base_image_index]

    # print(images.shape, features.shape)
    images = np.delete(images, base_image_index, axis=0)
    features = np.delete(features, base_image_index, axis=0)
    # print(images.shape, features.shape)
    
    # 特征匹配
    matches = match_features(features, base_image_feature)
    # print(matches.shape)

    obj_points = None
    obj_colors = None
    for i in range(len(images)):
        points3D, points2D = initialize_scene(base_image_feature[0], features[i][0], matches[i], K)
        rvec, tvec = solve_pnp_reconstruction(points3D, points2D, K)
        rvec, tvec, points3D = optimize_scene(rvec, tvec, points3D, points2D, K)
        colors = get_point_colors(base_image, points2D)
        
        if obj_points is None:
            obj_points = points3D
        else:
            obj_points = np.vstack((obj_points, points3D))

        if obj_colors is None:
            obj_colors = colors
        else:
            obj_colors = np.vstack((obj_colors, colors))
            

    # 将恢复的3D点转换为open3d的PointCloud对象并可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points)
    pcd.colors = o3d.utility.Vector3dVector(obj_colors / 255.0)  # 设置点云的颜色

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
    vis.close()

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(obj_points)
    # pcd.colors = o3d.utility.Vector3dVector(obj_colors / 255.0)  # 设置点云的颜色

    # # 使用 draw_geometries() 函数自动渲染成球形
    # o3d.visualization.draw_geometries([pcd], point_show_normal=False, mesh_show_wireframe=False)
