__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
import cv2
import scipy

class FeatureExtractor(object):
    def extract_feature(self, raw_feature):
        raise NotImplementedError


class TrainedFeatureExtractor(FeatureExtractor):
    def train(self):
        raise NotImplementedError

class SurfExtractor(FeatureExtractor):
    def __init__(self, surf_thresh=300, upright=False):
        self.surf_detector = cv2.SURF(surf_thresh, 4, 2, True, upright)

    def extract_feature(self, raw_feature):
        """
        :type raw_feature: scipy.array
        """
        keypoints, descriptors = self.surf_detector.detectAndCompute(raw_feature, None)
        return descriptors
