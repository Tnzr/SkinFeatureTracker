import cv2 as cv
import numpy as np


def sharp(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    dst = cv.filter2D(img, -1, kernel)

    return dst
