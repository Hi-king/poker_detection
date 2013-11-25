__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
import cv2
import scipy
class PokerDetector(object):
    def __init__(self, partial_frame_coordinates, vidid=0, word_num=100, surf_thresh=300, upright=False):
        self.pertial_frame_coordinates = partial_frame_coordinates
        self.vid = cv2.VideoCapture(vidid)
        self.frame = None
        self.partial_frames = [None]
        self.surf_detector = cv2.SURF(surf_thresh, 4, 2, True, upright)
        self.images = []
        self.features = []
        self.image_dict = {}
        self.feature_dict = {}
        self.word_num = word_num
        self.trained = False

    def update(self):
        status, self.frame = self.vid.read()
        self.partial_frames = [self.frame[0:20, 0:20]]
        #self.binarize()

    def binarize(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        retval, self.frame = cv2.threshold(self.frame, 128, 255, cv2.THRESH_BINARY)

    def extract_feature(self):
        keypoints, descriptors = self.surf_detector.detectAndCompute(self.frame, None)
        for keypoint in keypoints:
            center = map(int, keypoint.pt)
            cv2.circle(self.frame, tuple(center), 2, (255,0,0), -1)
        return descriptors

    def register(self, key=None):
        if key is None:
            key = -1
            while key in self.feature_dict:
                key -= 1

        if not key in self.feature_dict:
            self.feature_dict[key] = self.extract_feature()
            self.image_dict[key] = self.frame
        else:
            self.feature_dict[key] = scipy.concatenate((self.feature_dict[key], self.extract_feature()), axis=0)

    def _get_bof(self, descriptors):
        rawbof = scipy.zeros(self.word_num, dtype=scipy.float32)
        for feature in descriptors:
            feature = feature.reshape(1, feature.shape[0])
            status, results, neighbors, dists = self.feature_knn.find_nearest(feature, 1)
            rawbof[int(results[0])] += 1
        rawbof = rawbof.reshape(rawbof.shape[0], 1)
        return rawbof/scipy.sum(rawbof)

    def _train_bof(self):
        ## BOF
        status, labels, centers = cv2.kmeans(scipy.concatenate(tuple(self.features)), self.word_num,
                                             (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                             10, cv2.KMEANS_RANDOM_CENTERS)
        self.centers = centers
        classes = scipy.arange(len(centers), dtype=scipy.float32)
        self.feature_knn = cv2.KNearest()
        print self.centers.shape
        self.feature_knn.train(self.centers, classes)
        #self.feature_knn.train(self.centers, classes, 0, False, 1)
        self.bofs = scipy.concatenate(tuple([self._get_bof(feature) for feature in self.features]), axis=1)
        #self.bof_knn = cv2.KNearest(self.bofs, scipy.arange(self.K, dtype=scipy.float32), 0, False, 1)

        self.bofs = self.bofs.T


    def _decode_dict(self):
        dictkeys = self.feature_dict.keys()
        self.features = [self.feature_dict[key] for key in dictkeys]
        self.images = [self.image_dict[key] for key in dictkeys]

    def train(self):
        ## classifier
        self._decode_dict()
        self._train_bof()
        self.bof_knn = cv2.KNearest()
        self.bof_knn.train(self.bofs, scipy.arange(self.bofs.shape[0], dtype=scipy.float32))
        self.trained = True

    def classify(self, descriptors=None):
        if not descriptors: descriptors = self.extract_feature()
        status, results, neighbors, dists = self.bof_knn.find_nearest(self._get_bof(descriptors).T, 1)
        cv2.imshow("result", self.images[int(results[0])])

    def show(self):
        if self.trained: self.classify()
        cv2.imshow("raw", self.frame)
        for i in xrange(len(self.partial_frames)):
            cv2.imshow("im%i" % i, self.partial_frames[i])




import sklearn.ensemble
class RForestPokerDetector(PokerDetector):
    def __init__(self, partial_frame_coordinates, vidid=0, word_num=100):
        super(RForestPokerDetector, self).__init__(partial_frame_coordinates, vidid, word_num)
        self.labels = []

    def register(self, key=None):
        if key is None:
            key = -1
            while key in self.feature_dict:
                key -= 1

        if not key in self.feature_dict:
            self.feature_dict[key] = []
            self.image_dict[key] = self.frame
        self.feature_dict[key].append(self.extract_feature())

    def _decode_dict(self):
        dictkeys = self.feature_dict.keys()
        self.features = []
        self.labels = []
        for key in dictkeys:
            for feature in self.feature_dict[key]:
                self.features.append(feature)
                self.labels.append(key)
        #self.features = [feature for feature in self.feature_dict[key] for key in dictkeys]
        #self.images = [self.image_dict[key] for key in dictkeys]
        #self.labels = [key for feature in self.feature_dict[key] for key in dictkeys]

    def _train_bof(self):
        ## BOF
        status, labels, centers = cv2.kmeans(scipy.concatenate(tuple(self.features)), self.word_num,
                                             (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                             10, cv2.KMEANS_RANDOM_CENTERS)
        self.centers = centers
        classes = scipy.arange(len(centers), dtype=scipy.float32)
        self.feature_knn = cv2.KNearest()
        print self.centers.shape
        self.feature_knn.train(self.centers, classes)
        #self.feature_knn.train(self.centers, classes, 0, False, 1)
        self.bofs = scipy.concatenate(tuple([self._get_bof(feature) for feature in self.features]), axis=1)
        #self.bof_knn = cv2.KNearest(self.bofs, scipy.arange(self.K, dtype=scipy.float32), 0, False, 1)

        self.bofs = self.bofs.T

    def train(self):
        self._decode_dict()
        self._train_bof()
        self.classifier = sklearn.ensemble.RandomForestClassifier(100)
        print self.labels
        self.classifier.fit(self.bofs, scipy.array(self.labels))
        self.trained = True

    def classify(self, descriptors=None):
        if not descriptors: descriptors = self.extract_feature()
        results = self.classifier.predict(self._get_bof(descriptors).T)
        cv2.imshow("result", self.image_dict[int(results[0])])
