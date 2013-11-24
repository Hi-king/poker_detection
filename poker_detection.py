#!/usr/bin/env python
import cv2
from lib import PokerDetector

FRAMERATE = 10 #FPS

#vid = cv2.VideoCapture(1)
#vid.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
#vid.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
#if not vid.isOpened():
#    raise Exception("video not opened")


detector = PokerDetector([[0, 0, 20, 20]])
while True:
    detector.update()
    #vid.grab()
    #status, frame = vid.read()
    #cv2.imshow("test", frame)
    detector.show()
    cv2.waitKey(1000/FRAMERATE)