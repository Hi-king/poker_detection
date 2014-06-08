#!/usr/bin/env python
__author__ = 'hiking'
__email__ = 'hikingko1@gmail.com'

import cv2
import sys
import os
script_path = os.path.dirname(__file__)
script_path = script_path if len(script_path) else "."
sys.path.append(script_path+"/..")

from lib import RForestPokerDetector

WIDTH = 1000
HEIGHT = 500
MAIN_WINNAME = "poker detection"


if __name__=='__main__':
    window = cv2.namedWindow(MAIN_WINNAME)
    cv2.moveWindow(MAIN_WINNAME, 0, 0)
    detector = RForestPokerDetector([[0, 0, 20, 20]], 0, word_num=100, surf_thresh=50, upright=True)


    while True:
        detector.update()
        img = cv2.resize(detector.frame, (500, 500))
        cv2.imshow(MAIN_WINNAME, img)


