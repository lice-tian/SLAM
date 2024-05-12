# perform feature matching here

# return the matching result
import cv2
import numpy as np

def match_features(features):
    """
    """

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 暂时不清楚需要匹配哪几张图片
    all_matches = []
    for feature1, feature2 in zip(features, features[1:]):
        matches = flann.knnMatch(feature1[1], feature2[1], k=2)
        
        # good_matches = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good_matches.append(m)
        
        all_matches.append(matches)
        break
    
    all_matches = np.array(all_matches, dtype=object)

    return all_matches
        