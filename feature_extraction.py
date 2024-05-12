# perform feature extraction here
# return the feature vector
import numpy as np
import cv2

def extract_features(images):
    """
    """
    sift = cv2.SIFT_create()

    features = []
    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image.astype(np.uint8), None)
        features.append([keypoints, descriptors])
    features = np.array(features, dtype=object)

    return features    
