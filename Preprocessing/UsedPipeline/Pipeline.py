import logging

import cv2 as cv
import numpy as np
import torch.cuda


class Preprocessing:
    def __init__(self, img):
        self.img = img

    def sharpen(self):
        return cv.filter2D(cv.imread(self.img), -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    def histogram(self):
        return cv.equalizeHist(cv.cvtColor(cv.imread(self.img), cv.COLOR_BGR2GRAY))
