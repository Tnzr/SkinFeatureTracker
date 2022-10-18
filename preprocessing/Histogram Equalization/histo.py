from __future__ import print_function
import cv2 as cv
import argparse


def hist(img):
    src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(src)

    return dst
