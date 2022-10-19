import cv2 as cv


def hist(img):
    src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(src)

    return dst
