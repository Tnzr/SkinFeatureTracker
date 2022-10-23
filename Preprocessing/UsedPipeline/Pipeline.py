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

    def brighten(self, value=30):
        h, s, v = cv.split(cv.cvtColor(cv.imread(self.img), cv.COLOR_BGR2HSV))

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        return cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2BGR)

    def contrast(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        lab = cv.cvtColor(cv.imread(self.img), cv.COLOR_BGR2LAB)

        l_channel, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clip_limit, tile_grid_size)
        limg = cv.merge((clahe.apply(l_channel), a, b))

        return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    def bilateral(self, d=9, sigma_color=75, sigma_space=75):
        return cv.bilateralFilter(cv.imread(self.img), d, sigma_color, sigma_space)