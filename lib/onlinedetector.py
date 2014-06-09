__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
"""
DetectorBuilder takes trainingset and yield Detector
"""
from featureextractor import FeatureExtractor
import cv2

class Detector(object):
    def __init__(self, classifier, raw_extractors=[], trained_extractors=[]):
        """
        :type raw_extractors: [FeatureExtractor]
        :type trained_extractors: [FeatureExtractor]
        """
        self.raw_extractors = raw_extractors
        self.trained_extractors = trained_extractors
        self.detector = None

    def update(self, raw_feature):
        #for extractor in trained_extractors:
        raise NotImplementedError

    def get_classifier(self):
        """
        :return: Detector
        """
        raise NotImplementedError

    def train(self, raw_feature):
        raise NotImplementedError

    def classify(self):
        if self.detector is None: raise "should be trained before classify"
        raise NotImplementedError


class BagofFeaturesDetector(Detector):
    def __init__(self, raw_extractors=[]):
        self.raw_extractors = raw_extractors
        self.teacher_vectors = []

    def update(self, raw_feature, classid):
        self.teacher_vectors.append(reduce(lambda x,y: x+y, [extractor.extract_feature(raw_feature) for extractor in self.raw_extractors]))

    def train(self):
        raise NotImplementedError
