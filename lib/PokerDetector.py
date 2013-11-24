__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'
import cv2
class PokerDetector(object):
    def __init__(self, partial_frame_coordinates, vidid=1):
        self.pertial_frame_coordinates = partial_frame_coordinates
        self.vid = cv2.VideoCapture(vidid)
        self.frame = None
        self.partial_frames = [None]
        self.surf_detector = cv2.SURF()

    def update(self):
        status, self.frame = self.vid.read()
        self.partial_frames = [self.frame[0:20, 0:20]]

    def show(self):
        kp, detectors = self.surf_detector.detect(self.frame)

        cv2.imshow("raw", self.frame)
        for i in xrange(len(self.partial_frames)):
            cv2.imshow("im%i" % i, self.partial_frames[i])