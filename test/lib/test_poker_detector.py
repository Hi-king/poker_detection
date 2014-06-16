__author__ = 'keisuke_ogaki'
__author__ = 'keisuke_ogaki'
# -*- coding:utf-8 -*-
import unittest
import cv2
from poker_detector import PokerDetector

class PokerDetectorTest(unittest.TestCase):
    def test_recognize_as_same(self):
        classifier = PokerDetector()
        img1 = cv2.imread("../data/7s.JPG")
        img2 = cv2.imread("../data/10d.JPG")

        classifier.update(img1)
        classifier.update(img2)
        classifier.train()

        self.assertNotEqual(classifier.classify(img1), classifier.classify(img2))
