import cv2 as cv
import argparse
from histogram import hist
from sharpen import sharp


# Ugly use of literal files, but this is just an example to show imported functions work
def main():
    cv.imwrite('hist.png', hist(cv.imread('9.png')))
    cv.imwrite('hist_sharp.png', sharp(cv.imread('hist.png')))


if __name__ == "__main__":
    main()
