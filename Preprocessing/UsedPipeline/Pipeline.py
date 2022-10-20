import logging

import cv2 as cv
import numpy as np
import torch.cuda


class Preprocessing:
    def __init__(self, img):
        self.img = img

    def sharpen(self):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        src = cv.imread(self.img)
        shp = cv.filter2D(src, -1, kernel)

        return shp

    def histogram(self):
        src = cv.imread(self.img)
        gre = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        dst = cv.equalizeHist(gre)

        return dst
