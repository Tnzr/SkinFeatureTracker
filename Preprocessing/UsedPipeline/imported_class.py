import cv2 as cv
from Pipeline import Preprocessing


def main():
    PreProcClass = Preprocessing('9.png')
    cv.imwrite('hist.png', PreProcClass.histogram())
    cv.imwrite('sharp.png', PreProcClass.sharpen())
    cv.imwrite('brighten.png', PreProcClass.brighten())
    cv.imwrite('hist_sharp.png', Preprocessing('hist.png').sharpen())
    cv.imwrite('contrast.png', PreProcClass.contrast())
    cv.imwrite('bilateral.png', PreProcClass.bilateral())
    cv.imwrite('bilateral_custom.png', PreProcClass.bilateral(75, 75))


if __name__ == "__main__":
    main()
