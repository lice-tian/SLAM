# perform bundle adjustment here
import cv2
import numpy as np
from scipy.optimize import minimize

def reprojection_error(params, img_pts, K):
    rvec = params[0:3]
    tvec = params[3:6]
    pts = params[6:].reshape((-1, 3))

    # Project 3D points to 2D using current camera pose
    projected_pts, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
    projected_pts = projected_pts.squeeze()
    
    # Compute reprojection error
    error = np.linalg.norm(img_pts - projected_pts, axis=1)
    
    return np.mean(error)

def optimize_scene(rvec, tvec, points3D, points2D, K):
    initial_guess = np.zeros(6 + 3 * len(points3D))
    initial_guess[:3] = rvec.flatten()
    initial_guess[3:6] = tvec.flatten()
    initial_guess[6:] = points3D.flatten()

    # Optimize using scipy
    res = minimize(reprojection_error, initial_guess, args=(points2D, K), method='L-BFGS-B')
    optimized_params = res.x

    # Extract optimized camera pose and 3D points
    rvec_optimized = optimized_params[:3]
    tvec_optimized = optimized_params[3:6]
    optimized_points = optimized_params[6:].reshape((-1, 3))

    return rvec_optimized, tvec_optimized, optimized_points
    