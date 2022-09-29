"""
Applies a Gaussian Filter followed by Laplacian Edge Detection
Used to enhance skin features and make them more detectable
Findings:
Effective for front lit facial features
Ineffective in uneven lighting
Ineffective for areas with 'noise' (hairy forearms are noisy forearms)
"""
from itertools import product
import cv2
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
import numpy
import numpy as np
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros

def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return g


def gaussian_filter(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
    image_array = zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = ravel(image[i : i + k_size, j : j + k_size])
        image_array[row, :] = window
        row += 1

    #  turn the kernel into shape(k*k, 1)
    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = ravel(gaussian_kernel)

    # reshape and get the dst image
    dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return dst

# Filters for calculating Laplacian(uncomment the one you want)

'''
# works... okay... with arms
conv_kernel = np.array([[-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, 24, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1]])

'''

conv_kernel = np.array([[0,1,0],
                        [1,-4,1],
                        [0,1,0]])

conv_kernel = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])


# Function to calculate 2D convolution of two matrix
def conv2d(led, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = led.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(led[i:i + m, j:j + m] * kernel)
    return new_image
  

if __name__ == "__main__":
    # read original image
    img = imread(r"front_nobg_moles.jpg")
    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    gaussian3x3 = gaussian_filter(gray, 3, sigma=1)
    # works well with faces
    gaussian5x5 = gaussian_filter(gray, 9, sigma=2)
    # works... okay... with arms
    # gaussian5x5 = gaussian_filter(gray, 3, sigma=1)

    cv2.imwrite('_GF.jpg', gaussian5x5)
    led = cv2.imread('_GF.jpg', 0)
    cv2.imwrite('_GF_LED.jpg', conv2d(led, conv_kernel))
    waitKey()
