__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
"""
DetectorBuilder takes trainingset and yield Detector
"""
from featureextractor import FeatureExtractor
import cv2
import scipy
import sklearn.ensemble
import sklearn.cluster


class Detector(object):
    def train(self):
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


class RFDetector(Detector):
    class _RFDetector(StaticDetector):
        def __init__(self, raw_features, labels, n_estimators=100):
            self.classifier = None
            self.train(raw_features, labels, n_estimators)

        def train(self, raw_features, labels, n_estimators=100):
            ns, nf = scipy.concatenate(raw_features, axis=1).T.shape
            #scipy.array(labels).shape
            self.classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)
            self.classifier.fit(scipy.concatenate(raw_features, axis=1).T, scipy.array(labels))

        def classify(self, raw_feature):
            results = self.classifier.predict(raw_feature)
            return results[0]


    def __init__(self, raw_extractor):
        """
        :type raw_extractor: FeatureExtractor
        """
        self.raw_extractor = raw_extractor
        self.teacher_vectors = []
        self.raw_vectors = []
        self.classifier = None
        self.key_dict = {}
        self.id_dict = []
        self.labels = []

    def _set_key(self, key):
        if self.key_dict.has_key(key):
            return self.key_dict[key]
        else:
            self.id_dict.append(key)
            self.key_dict[key] = len(self.id_dict)-1
            return self.key_dict[key]

    def update(self, raw_feature, key):
        """
        :type raw_feature: [float]
        :type key: str
        """
        self.raw_extractor.update(raw_feature, key)
        #self.teacher_vectors.append(self._extract_raw_feature(raw_feature))
        self.raw_vectors.append(raw_feature)
        self.labels.append(self._set_key(key))

    def train(self):
        print "train"
        for vector, key in zip(self.raw_vectors, self.labels):
            self.raw_extractor.update(vector, key)
        self.raw_extractor.train()
        for vector in self.raw_vectors:
            self.teacher_vectors.append(self._extract_raw_feature(vector))
        self.classifier = self._RFDetector(self.teacher_vectors, self.labels)

    def classify(self, raw_feature):
        if self.classifier is None:
            raise Exception("should be trained before classify")
        return self.id_dict[self.classifier.classify(self._extract_raw_feature(raw_feature).T)]

    def _extract_raw_feature(self, raw_feature):
        """
        :raw_feature: [float]
        :return: [[float]]
        """
        return self.raw_extractor.extract_feature(raw_feature)


class RFProbabilityDetector(RFDetector):
    def classify(self, raw_feature):
        #TODO: return probabilitydict
        raise NotImplementedError


class BagofFeaturesDetector(OnlineDetector):
    class KmeansClassifier(StaticDetector):
        def __init__(self, raw_features, word_num=10):
            self.word_num = word_num
            self.train(raw_features)

        def train(self, raw_features):
            #print scipy.array(raw_features).dims
            #print scipy.array(raw_features, dtype=scipy.float32)
            # status, labels, centroids = cv2.kmeans(scipy.array(raw_features, dtype=scipy.float32),
            #                                      self.word_num,
            #                                      (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            #                                      10,
            #                                      cv2.KMEANS_RANDOM_CENTERS)
            # print centroids
            # classes = scipy.arange(len(centroids), dtype=scipy.float32)
            # self.feature_knn = cv2.KNearest()
            # self.feature_knn.train(centroids, classes)
            self.classifier = sklearn.cluster.KMeans(self.word_num)
            self.classifier.fit(raw_features)

        def classify(self, raw_feature):
            # status, results, neighbors, dists = self.feature_knn.find_nearest(scipy.matrix(raw_feature, dtype=scipy.float32), 1)
            # print results
            result = self.classifier.predict(raw_feature)
            return result

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
        self.teacher_vectors += self._extract_raw_feature(raw_feature)

    def train(self):
        print "train"
        self.classifier = BagofFeaturesDetector.KmeansClassifier(self.teacher_vectors, word_num=self.word_num)
        print "done"

    def extract_feature(self, raw_feature):
        print "extract_feature %d" % len(self._extract_raw_feature(raw_feature))
        if self.classifier is None:
            raise Exception("should be trained before classify")
        rawbof = scipy.zeros(self.word_num, dtype=scipy.float32)
        print "go   "
        for each_feature in self._extract_raw_feature(raw_feature):
            classid = self.classifier.classify(self._extract_raw_feature(raw_feature))
            rawbof[classid] += 1.0
            print rawbof
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



