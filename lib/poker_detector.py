__author__ = 'keisuke_ogaki'
__email__ = 'hikingko1@gmail.com'

import featureextractor
import onlinedetector

class PokerDetector(onlinedetector.Detector):
    def __init__(self, word_num, surf_thresh=100, n_jobs=4):
        SURF_extractor = featureextractor.SurfExtractor(surf_thresh=surf_thresh)
        BoF_extractor = onlinedetector.BagofFeaturesDetector(SURF_extractor, word_num=word_num, n_jobs=n_jobs)
        RF_classifier = onlinedetector.RFDetector(BoF_extractor, n_estimators=1000)
        self.classifier = RF_classifier

    def update(self, raw_img, key):
        self.classifier.update(raw_img, key)

    def train(self):
        self.classifier.train()

    def classify(self, raw_img):
        return self.classifier.classify(raw_img)

    def classify_proba(self, raw_img):
        return self.classifier.classify_proba(raw_img)
