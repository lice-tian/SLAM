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



def read_images(image_folder):
    image_files = glob.glob(os.path.join(images_folder, '*.png'))

    images = []
    for file_path in image_files:
        # 读入灰度图像
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        images.append(image)
    images = np.array(images, dtype=np.float32)

    return images


if __name__ == '__main__':
    images_folder = 'images'
    images = read_images(images_folder)


    # 初始化特征点和匹配点集合
    features = []
    matches = []

    # SIFT特征提取
    features = extract_features(images)

    # 特征匹配
    matches = match_features(features)

    # 场景初始化
    # 假设我们已经从matches_list中提取了前两张图像的匹配点
    first_image_matches = matches[0]
    first_keypoints = features[0][0]
    second_keypoints = features[1][0]

    # 调用initialize_scene函数进行场景初始化
    # 这里需要传入匹配的特征点和对应的关键点
    camera_pose = initialize_scene(first_keypoints, second_keypoints, first_image_matches)