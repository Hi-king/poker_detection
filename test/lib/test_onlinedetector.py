__author__ = 'keisuke_ogaki'
# -*- coding:utf-8 -*-
import unittest
from onlinedetector import BagofFeaturesDetector
from featureextractor import FeatureExtractor

class DummyExtractor(FeatureExtractor):
    def extract_feature(self, raw_feature):
        return raw_feature

class BagofFeaturesDetectorTest(unittest.TestCase):
    def test_train_and_classify(self):
        detector = BagofFeaturesDetector(raw_extractors=[DummyExtractor()])
        detector.update(1, 1)
        detector.update(2, 2)
        detector.train()
        self.assertEqual(detector.classify(1), 1)
