__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
import cv2
import scipy
class PokerDetector(object):
    def __init__(self, partial_frame_coordinates, vidid=0, word_num=100):
        self.pertial_frame_coordinates = partial_frame_coordinates
        self.vid = cv2.VideoCapture(vidid)
        self.frame = None
        self.partial_frames = [None]
        self.surf_detector = cv2.SURF()
        self.images = []
        self.features = []
        self.word_num = word_num

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

    def register(self):
        self.images.append(self.frame)
        self.features.append(self.extract_feature())
        print self.features

    def _get_bof(self, descriptors):
        rawbof = scipy.zeros(self.word_num, dtype=scipy.float32)
        for feature in descriptors:
            feature = feature.reshape(1, feature.shape[0])
            status, results, neighbors, dists = self.feature_knn.find_nearest(feature, 1)
            rawbof[int(results[0])] += 1
        rawbof = rawbof.reshape(rawbof.shape[0], 1)
        return rawbof/scipy.sum(rawbof)


    def train(self):
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
        self.bof_knn = cv2.KNearest()
        self.bof_knn.train(self.bofs, scipy.arange(self.bofs.shape[0], dtype=scipy.float32))

    def classify(self, descriptors=None):
        if not descriptors: descriptors = self.extract_feature()
        status, results, neighbors, dists = self.bof_knn.find_nearest(self._get_bof(descriptors).T, 1)
        cv2.imshow("result", self.images[int(results[0])])

    def show(self):
        cv2.imshow("raw", self.frame)
        for i in xrange(len(self.partial_frames)):
            cv2.imshow("im%i" % i, self.partial_frames[i])