import cv2 as cv
from Pipeline import Preprocessing


def main():
    PreProcClass = Preprocessing('9.png')

    HistogramObject = PreProcClass.histogram()
    cv.imwrite('hist.png', HistogramObject)

    SharpenObject = PreProcClass.sharpen()
    cv.imwrite('sharp.png', SharpenObject)

    PreProcClass = Preprocessing('hist.png')
    HistogramSharpen = PreProcClass.sharpen()
    cv.imwrite('hist_sharp.png', HistogramSharpen)


if __name__ == "__main__":
    main()
