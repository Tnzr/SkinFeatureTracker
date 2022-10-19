import cv2
import numpy as np
import math

gray = cv2.imread('arm_out5x5.jpg')
color = cv2.imread('arm_5x5_HM.jpg')

# Logarithmic Transformation 

def logtrans(img,c):
    limgf = np.zeros(img.shape)
    img[img==0] = 1
    limgf = c * (np.log(img))
    limgf[limgf>255] = 255
    limgf = limgf.astype(int)
    return limgf

c2 = 45 #Select this variable as per your preference between 0-255

x = logtrans(gray,c2)
cv2.imwrite('1new_image_g.jpg',x)


y = logtrans(color,c2)
cv2.imwrite('1new_image_c.jpg',y)

