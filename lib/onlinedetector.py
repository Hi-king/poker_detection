__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
"""
DetectorBuilder takes trainingset and yield Detector
"""
from featureextractor import FeatureExtractor

class DetectorBuilder(object):
    def __init__(self, classifier, raw_extractors=[], trained_extractors=[]):
        """
        :type raw_extractors: [FeatureExtractor]
        :type trained_extractors: [FeatureExtractor]
        """
        self.raw_extractors = raw_extractors
        self.trained_extractors = trained_extractors

    def update_trainset(self, raw_feature):
        raise NotImplementedError

    def get_classifier(self):
        """
        :return: Detector
        """
        raise NotImplementedError

class Detector(object):
    def extract_feature(self, raw_feature):
        return map(lambda x: x.extract_feature(raw_feature), self.raw_extractors+self.trained_extractors)

