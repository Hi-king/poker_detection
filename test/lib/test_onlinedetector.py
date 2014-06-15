__author__ = 'keisuke_ogaki'
# -*- coding:utf-8 -*-
import unittest
from onlinedetector import BagofFeaturesDetector
from featureextractor import FeatureExtractor
import scipy

class DummyExtractor(FeatureExtractor):
    def extract_feature(self, raw_feature):
        return [raw_feature]

class BagofFeaturesDetectorTest(unittest.TestCase):
    def test_train_and_classify(self):
        detector = BagofFeaturesDetector(word_num=2, raw_extractor=DummyExtractor())
        detector.update([1], 1)
        detector.update([2], 2)
        detector.train()
        print detector.extract_feature([1])
        print detector.extract_feature([2])
        self.assertNotEqual(detector.extract_feature([1])[0], detector.extract_feature([2])[0])
        self.assertNotEqual(detector.extract_feature([1])[1], detector.extract_feature([2])[1])
