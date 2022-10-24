import cv2 as cv
import numpy as np


class Preprocessing:
    def __init__(self, img=None):
        self.img = img
    @staticmethod
    def sharpen(img):
        return cv.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    def histogram(self):
        return cv.equalizeHist(cv.cvtColor(cv.imread(self.img), cv.COLOR_BGR2GRAY))

    def brighten(self, value=30):
        h, s, v = cv.split(cv.cvtColor(cv.imread(self.img), cv.COLOR_BGR2HSV))

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        return cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2BGR)

    @staticmethod
    def contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        color = len(img.shape) == 3
        clahe = cv.createCLAHE(clip_limit, tile_grid_size)

        if color:
            lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            l_channel, a, b = cv.split(lab)
            clashed = clahe.apply(l_channel)
            limg = cv.merge((clashed, a, b))
        else:
            clashed = clahe.apply(img)

        return cv.cvtColor(limg, cv.COLOR_LAB2BGR) if color else clashed

    def bilateral(self, d=9, sigma_color=75, sigma_space=75):
        return cv.bilateralFilter(cv.imread(self.img), d, sigma_color, sigma_space)

    def run(self, img):
        out = self.contrast(img)
        out = self.sharpen(out)
        return out

# where is the pipeline???
