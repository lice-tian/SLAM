# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points

import numpy as np
import cv2

def initialize_scene(keypoints1, keypoints2, matches):
    # 初步筛选匹配点
    good = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts1.append(keypoints1[m.queryIdx].pt)
            pts2.append(keypoints2[m.queryIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # 相机内参矩阵
    K = np.loadtxt('camera_intrinsic.txt')

    # 计算本质矩阵 E
    E, _ = cv2.findEssentialMat(pts1, pts2, K)

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # 三角测量
    projMatr1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])    # 第一个相机参数
    projMatr2 = np.concatenate((R, t), axis=1)               # 第二个相机参数
    projMatr1 = np.matmul(K, projMatr1) # 相机内参 相机外参
    projMatr2 = np.matmul(K, projMatr2)
    points4D = cv2.triangulatePoints(projMatr1, projMatr2, pts1.T, pts2.T)
    points4D /= points4D[3]       # 归一化
    points3D = points4D.T[:,0:3]

    return points3D,pts2
