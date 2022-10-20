import cv2 as cv
from Pipeline import Preprocessing


def main():
    PreProcClass = Preprocessing('9.png')
    cv.imwrite('hist.png', PreProcClass.histogram())
    cv.imwrite('sharp.png', PreProcClass.sharpen())
    cv.imwrite('hist_sharp.png', Preprocessing('hist.png').sharpen())


if __name__ == "__main__":
    main()
