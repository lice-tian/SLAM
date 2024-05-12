# perform complete 3D reconstruction from 11 images
import os
import glob
import cv2
import numpy as np
from feature_extraction import extract_features
from feature_matching import match_features


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

    # SIFT特征提取
    features = extract_features(images)

    # 特征匹配
    matches = match_features(features)