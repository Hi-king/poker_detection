#!/usr/bin/env python
import cv2
import sys
import os
script_path = os.path.dirname(__file__)
script_path = script_path if len(script_path) else "."
sys.path.append(script_path+"/..")
from lib import PokerDetector
from lib import RForestPokerDetector

FRAMERATE = 10 #FPS
#detector = PokerDetector([[0, 0, 20, 20]], 0, word_num=50, surf_thresh=300, upright=True)
detector = RForestPokerDetector([[0, 0, 20, 20]], 0, word_num=1000, surf_thresh=50, upright=False)
while True:
    detector.update()
    #vid.grab()
    #status, frame = vid.read()
    
    #cv2.imshow("test", frame)
    detector.show()
    key = cv2.waitKey(1000/FRAMERATE)
    print key
    if key == -1:
        continue
    elif key == ord('s'):
        detector.register()
    elif key == ord('t'):
        detector.train()
    elif key == ord('c'):
        detector.classify()
    else:
        detector.register(key)
