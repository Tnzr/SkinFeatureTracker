import cv2
import numpy as np
import logging


class TextureEnhancer:

    ENHANCER_HISTEQ = 1
    ENHANCER_CLASHE = 2
    INTERPOLATION_SHARP = 13

    def __init__(self, sharpening_ksize=3):
        self.sharpening_ksize = sharpening_ksize

    @staticmethod
    def gen_sharp_kernel(k_size: int = 3):
        if k_size == 1:
            logging.warning(f"Kernel size 1 is not valid, kernel must be 3x3 minimum. Using a 3x3 kernel instead.")
            k_size = 3
        if not (k_size % 2):
            logging.warning(f"Kernel size {k_size}x{k_size} was requested but sharpening kernel must have odd dimensions. Kernel size {k_size + 1}x{k_size + 1} will be used")
            k_size += 1
        kernel = np.ones((k_size, k_size))
        mid = k_size // 2
        peak = np.sum(kernel)
        kernel *= -1
        kernel[mid, mid] = peak
        return kernel

    def sharpen(self, img, k_size=5):
        kernel = self.gen_sharp_kernel(k_size)
        return cv2.filter2D(img, -1, kernel)

    def histogram_eq(self, img):
        color = len(img.shape) == 3

        if color:
            out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = cv2.equalizeHist(out if color else img)

        return out

    def brighten(self, img, value=30):
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    @staticmethod
    def clashe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        color = len(img.shape) == 3
        clahe = cv2.createCLAHE(clip_limit, tile_grid_size)

        if color:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clashed = clahe.apply(l_channel)
            limg = cv2.merge((clashed, a, b))
        else:
            clashed = clahe.apply(img)

        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) if color else clashed

    def bilateral(self, img, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def run(self, img, enhancer=ENHANCER_CLASHE, interpolation=INTERPOLATION_SHARP):
        if enhancer not in [self.ENHANCER_HISTEQ, self.ENHANCER_CLASHE]:
            logging.warning(f"Unknown Texture Enhancement code {enhancer} was inputted. Defaulting to ENHANCER_CLASHE")
            enhancer = self.ENHANCER_CLASHE
        if interpolation not in [self.INTERPOLATION_SHARP]:
            logging.warning(f"Unknown Interpolation code {interpolation} was inputted. Defaulting to INTERPOLATION_SHARP")
            interpolation = self.INTERPOLATION_SHARP

        if enhancer == self.ENHANCER_CLASHE:
            out = self.clashe(img)
        elif enhancer == self.ENHANCER_HISTEQ:
            out = self.histogram_eq(img)

        if interpolation == self.INTERPOLATION_SHARP:
            out = self.sharpen(out, self.sharpening_ksize)

        return out


if __name__ == '__main__':
    PreProcClass = TextureEnhancer()
    image = cv2.imread('9.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('pipeline_run.png', PreProcClass.run(gray))
    cv2.imwrite('hist.png', PreProcClass.histogram_eq(image))
    cv2.imwrite('sharp.png', PreProcClass.sharpen(image))
    cv2.imwrite('sharp_custom_7.png', PreProcClass.sharpen(image, 7))
    cv2.imwrite('brighten.png', PreProcClass.brighten(image))
    cv2.imwrite('brighten_100.png', PreProcClass.brighten(image, 100))
    cv2.imwrite('clashe.png', PreProcClass.clashe(image))
    cv2.imwrite('clashe_custom.png', PreProcClass.clashe(image, 4.0, (16, 16)))
    cv2.imwrite('bilateral.png', PreProcClass.bilateral(image))
    cv2.imwrite('bilateral_75.png', PreProcClass.bilateral(image, 75, 75))
