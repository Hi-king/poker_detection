#!/usr/bin/env python
__author__ = 'keisuke_ogaki'
import cv2
import sys
import os
import scipy
import argparse
import collections
script_path = os.path.dirname(__file__)
script_path = script_path if len(script_path) else "."
sys.path.append(script_path+"/..")
from lib import RForestPokerDetector, RForestDensityPokerDetector
from lib import PokerDetector

DELIMITER = ","

parser = argparse.ArgumentParser()
parser.add_argument('train_tsv_filename', help='[imgfilename, classname]')
parser.add_argument('test_filename', help='[imgfilename]')

if __name__=='__main__':
    args = parser.parse_args()
    train_list = [line.rstrip().split(DELIMITER) for line in open(args.train_tsv_filename)]
    test_list = [line.rstrip().split(DELIMITER) for line in open(args.test_filename)]

    #detector = RForestPokerDetector([[0, 0, 20, 20]], 0, word_num=1000, surf_thresh=50, upright=False)
    #suite_detector = RForestPokerDetector([[0, 0, 20, 20]], 0, word_num=1000, surf_thresh=50, upright=False)
    #number_detector = RForestDensityPokerDetector([[0, 0, 20, 20]], 0, word_num=1000, surf_thresh=50, upright=False)
    number_detector = PokerDetector(word_num=100, surf_thresh=100)

    ##train
    classnames = []
    suite_classnames = []
    number_classnames = []
    print train_list
    #for imgfilename, classname in train_list:
    #    if not classname in classnames: classnames.append(classname)
    #    classid = classnames.index(classname)
    #    img = cv2.imread(imgfilename)
    #    detector.update(img)
    #    detector.register(classid)
    #    print(imgfilename, classid, classname)

    #for imgfilename, classname in train_list:
    #    classname = classname[-1]
    #    if not classname in suite_classnames: suite_classnames.append(classname)
    #    classid = suite_classnames.index(classname)
    #    img = cv2.imread(imgfilename)
    #    suite_detector.update(img)
    #    suite_detector.register(classid)
    #    print(imgfilename, classid, classname)
    #

    #subrect = [(0, 0), (0, 0)]
    def on_mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["subrect"][0] = (x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            param["subrect"][1] = (x, y)
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDOWN:
            param["img"] = param["defaultimg"].copy()
            cv2.rectangle(param["img"], *param["subrect"], color=(255, 0, 0))
            cv2.imshow("tmp", param["img"])

    # img = cv2.imread(train_list[0][0])
    # cv2.imshow("test", img)
    # binarized_img = cv2.adaptiveThreshold(
    #     cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY),
    #     1,
    #     cv2.cv.CV_ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.cv.CV_THRESH_BINARY,
    #     111,
    #     5)
    #
    # gray_img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    # s, binarized_img = cv2.threshold(
    #     gray_img,
    #     float(gray_img.max())/2,
    #     255,
    #     cv2.cv.CV_THRESH_BINARY
    # )
    # print binarized_img
    # cv2.imshow("bin", binarized_img)
    # cv2.waitKey(-1)
    # print dir(binarized_img)
    # print binarized_img.size
    # print scipy.sum(binarized_img)/binarized_img.size
    # exit(1)

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


    img = cv2.imread(train_list[0][0])
    subrect = calib_rect(img)
    for imgfilename, classname in train_list:
        classname = classname[:-1]
        if not classname in number_classnames: number_classnames.append(classname)
        classid = number_classnames.index(classname)
        img = cv2.imread(imgfilename)
        img = img[
            subrect[0][1]:subrect[1][1],
            subrect[0][0]:subrect[1][0]
        ]
        cv2.imshow("tmp", img)
        cv2.waitKey(1)
        #number_detector.update(img, subrect)
        number_detector.update(img, classid)
        #number_detector.register(classid)
        print(imgfilename, classid, classname)

    number_detector.train()
    ## test
    img = cv2.imread(test_list[100][0])
    subrect = calib_rect(img)
    for imgfilename, trueclassname in test_list:
        img = cv2.imread(imgfilename)
        img = img[
            subrect[0][1]:subrect[1][1],
            subrect[0][0]:subrect[1][0]
        ]
        cv2.imshow("tmp", img)
        # img = img[
        #     subrect[0][1]:subrect[1][1],
        #     subrect[0][0]:subrect[1][0]
        # ]
        #number_detector.update(img, subrect)
        #number_detector.update(img, subrect)
        #classid = detector.classify(to_show=False)
        #suite_classid = suite_detector.classify()
        number_classid = number_detector.classify(img)
        #print(imgfilename, classnames[classid], trueclassname)
        #print(imgfilename, suite_classnames[suite_classid], trueclassname)
        print(imgfilename, number_classnames[number_classid], trueclassname)
