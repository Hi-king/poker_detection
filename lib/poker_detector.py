__author__ = 'keisuke_ogaki'
__email__ = 'hikingko1@gmail.com'

import featureextractor
import onlinedetector

class PokerDetector(onlinedetector.Detector):
    def __init__(self, word_num):
        SURF_extractor = featureextractor.SurfExtractor()
        BoF_extractor = onlinedetector.BagofFeaturesDetector(SURF_extractor, word_num=word_num)
        RF_classifier = onlinedetector.RFDetector(BoF_extractor)
        self.classifier = RF_classifier

    def update(self, raw_img, key):
        self.classifier.update(raw_img, key)

    def train(self):
        self.classifier.train()

    def classify(self, raw_img):
        return self.classifier.classify(raw_img)
