#!/usr/bin/env python
import cv2
import sys
import os
import argparse
script_path = os.path.dirname(__file__)
script_path = script_path if len(script_path) else "."
sys.path.append(script_path+"/..")
from lib import PokerDetector

parser = argparse.ArgumentParser()
parser.add_argument('train_tsv_filename', help='[imgfilename, classname]')
parser.add_argument('--deviceid', type=int, default=0)

def on_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["subrect"][0] = (x, y)
    if event == cv2.EVENT_RBUTTONDOWN:
        param["subrect"][1] = (x, y)
    if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDOWN:
        param["img"] = param["defaultimg"].copy()
        cv2.rectangle(param["img"], *param["subrect"], color=(255, 0, 0))
        cv2.imshow("tmp", param["img"])

def calib_rect(img):
    width, height = img.shape[:2]
    subrect = [(0, 0), (height, width)]
    cv2.imshow("tmp", img)
    param = {}
    param["img"] = img
    param["defaultimg"] = img.copy()
    param["subrect"] = subrect
    cv2.setMouseCallback("tmp", on_mouse_event, param)
    while True:
        key = cv2.waitKey(-1)
        print key
        if key == 13: break
    print img.shape
    img = img[
        param["subrect"][0][1]:param["subrect"][1][1],
        param["subrect"][0][0]:param["subrect"][1][0]
    ]
    print param["subrect"]
    print img.shape
    cv2.imshow("tmp", img)
    cv2.waitKey(-1)
    return param["subrect"]


FRAMERATE = 10 #FPS
SHOWRATE = 10  # x FRAMERATE
DELIMITER = ","

if __name__=='__main__':
    # init
    args = parser.parse_args()
    train_list = [line.rstrip().split(DELIMITER) for line in open(args.train_tsv_filename)]
    number_detector = PokerDetector(word_num=1000, surf_thresh=100)
    suite_detector = PokerDetector(word_num=1000, surf_thresh=100)
    classnames = []
    suite_classnames = []
    number_classnames = []
    video = cv2.VideoCapture(args.deviceid)

    # train
    img = cv2.imread(train_list[0][0])
    subrect = calib_rect(img)
    for imgfilename, classname in train_list:
        numname = classname[:-1]
        suitename = classname[-1]
        #if not classname in number_classnames: number_classnames.append(classname)
        #classid = number_classnames.index(classname)
        img = cv2.imread(imgfilename)
        img = img[
            subrect[0][1]:subrect[1][1],
            subrect[0][0]:subrect[1][0]
        ]
        cv2.imshow("tmp", img)
        cv2.waitKey(1)
        number_detector.update(img, numname)
        suite_detector.update(img, suitename)
        #print(imgfilename, classname, classname)
    number_detector.train()
    suite_detector.train()

    # test
    while True:
        img = video.read()[1]
        key = cv2.waitKey(1)
        cv2.imshow("tmp", img)
        if key == 13:
            break
    subrect = calib_rect(img)
    i = 0
    last_num = "null"

    while True:
        i+=1
        cv2.waitKey(1000/FRAMERATE)
        img = video.read()[1]
        img = img[
            subrect[0][1]:subrect[1][1],
            subrect[0][0]:subrect[1][0]
        ]
        cv2.imshow("tmp", img)
        num, num_proba = number_detector.classify_proba(img)
        suite, suite_proba = suite_detector.classify_proba(img)
        print "\t".join(map(str, [num, suite, num_proba, suite_proba]))

