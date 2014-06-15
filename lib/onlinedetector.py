__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
"""
DetectorBuilder takes trainingset and yield Detector
"""
from featureextractor import FeatureExtractor
import cv2
import scipy

class Detector(object):
    def train(self, raw_feature):
        raise NotImplementedError

    def classify(self):
        if self.detector is None: raise "should be trained before classify"
        raise NotImplementedError

class OnlineDetector(Detector):
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

class StaticDetector(Detector):
    def train(self, raw_features):
        raise NotImplementedError

class BagofFeaturesDetector(OnlineDetector):
    class KmeansClassifier(StaticDetector):
        def __init__(self, word_num=10):
            self.word_num = word_num

        def train(self, raw_features):
            #print scipy.array(raw_features).dims
            print scipy.array(raw_features, dtype=scipy.float32)
            status, labels, centroids = cv2.kmeans(scipy.array(raw_features, dtype=scipy.float32),
                                                 self.word_num,
                                                 (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                                 10,
                                                 cv2.KMEANS_RANDOM_CENTERS)
            classes = scipy.arange(len(centroids), dtype=scipy.float32)
            self.feature_knn = cv2.KNearest()
            self.feature_knn.train(centroids, classes)

        def classify(self, raw_feature):
            print scipy.array(raw_feature)
            status, results, neighbors, dists = self.feature_knn.find_nearest(scipy.matrix(raw_feature, dtype=scipy.float32), 1)
            return int(results[0])

    def __init__(self, raw_extractor, word_num=10):
        """
        :type raw_extractor: FeatureExtractor
        """
        self.raw_extractor = raw_extractor
        self.teacher_vectors = []
        self.word_num = word_num
        self.classifier = None

    def update(self, raw_feature, classid):
        """
        :raw_feature: [float]
        """
        print self._extract_raw_feature(raw_feature)
        self.teacher_vectors += self._extract_raw_feature(raw_feature)

    def train(self):
        print "train"
        self.classifier = BagofFeaturesDetector.KmeansClassifier(self.word_num)
        self.classifier.train(self.teacher_vectors)

    def extract_feature(self, raw_feature):
        rawbof = scipy.zeros(self.word_num, dtype=scipy.float32)
        for each_feature in self._extract_raw_feature(raw_feature):
            classid = self.classifier.classify(self._extract_raw_feature(raw_feature))
            rawbof[classid] += 1
        rawbof = rawbof.reshape(rawbof.shape[0], 1)
        return rawbof/scipy.sum(rawbof)

    def _extract_raw_feature(self, raw_feature):
        """
        :raw_feature: [float]
        :return: [[float]]
        """
        return self.raw_extractor.extract_feature(raw_feature)

    #def classify(self, raw_feature):
    #    if self.classifier is None:
    #        raise Exception("should be trained before classify")
    #    return self.classifier.classify(self._extract_raw_feature(raw_feature))



