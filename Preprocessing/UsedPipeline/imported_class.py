import cv2 as cv
from TextureEnhancer import TextureEnhancer


def main():
    PreProcClass = TextureEnhancer('9.png')
    cv.imwrite('hist.png', PreProcClass.histogram())
    cv.imwrite('sharp.png', PreProcClass.sharpen())
    cv.imwrite('brighten.png', PreProcClass.brighten())
    cv.imwrite('hist_sharp.png', TextureEnhancer('hist.png').sharpen())
    cv.imwrite('contrast.png', PreProcClass.clashe())
    cv.imwrite('bilateral.png', PreProcClass.bilateral())
    cv.imwrite('bilateral_custom.png', PreProcClass.bilateral(75, 75))


if __name__ == "__main__":
    main()
