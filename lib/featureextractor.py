__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'


class FeatureExtractor(object):
    def extract_feature(self, raw_feature):
        raise NotImplementedError


class TrainedFeatureExtractor(FeatureExtractor):
    def train(self):
        raise NotImplementedError