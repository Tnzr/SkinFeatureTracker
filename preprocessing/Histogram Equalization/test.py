from __future__ import print_function
import cv2 as cv
import argparse
from histo import hist


def main():
    cv.imwrite('9_out.png', hist(cv.imread('9.png')))


if __name__ == "__main__":
    main()
