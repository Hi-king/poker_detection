__author__ = 'keisuke_ogaki'
# -*- coding:utf-8 -*-
import unittest
from onlinedetector import BagofFeaturesDetector
from onlinedetector import RFDetector
from onlinedetector import DetectorCombinator
from featureextractor import FeatureExtractor
import scipy
import numpy.testing


class DummyExtractor(FeatureExtractor):
    def train(self):
        pass

    def update(self, raw_feature, key):
        pass

    def extract_feature(self, raw_feature):
        return scipy.array([raw_feature])


class BagofFeaturesDetectorTest(unittest.TestCase):
    def test_train_and_classify(self):
        detector = BagofFeaturesDetector(word_num=2, raw_extractor=DummyExtractor())
        detector.update([1], 1)
        detector.update([2], 2)
        print "BOF train ..."
        detector.train()
        print "BOF train done"
        print detector.extract_feature([1])
        print detector.extract_feature([2])
        self.assertNotEqual(detector.extract_feature([1])[0], detector.extract_feature([2])[0])
        self.assertNotEqual(detector.extract_feature([1])[1], detector.extract_feature([2])[1])


class RFDetectorTest(unittest.TestCase):
    def test_single_chain(self):
        detector = RFDetector(raw_extractor=DummyExtractor())
        detector.update([1], "A")
        detector.update([2], "B")
        detector.train()
        self.assertEqual(detector.classify([0.9]), "A")


class DetectorCombinatorTest(unittest.TestCase):
    def test_combinator_extract(self):
        extractor1 = DummyExtractor()
        extractor2 = DummyExtractor()
        combined_extractor = DetectorCombinator([extractor1, extractor2])
        numpy.testing.assert_array_equal(combined_extractor.extract_feature([1,2]), scipy.array([1,2,1,2])[:, scipy.newaxis].T)
