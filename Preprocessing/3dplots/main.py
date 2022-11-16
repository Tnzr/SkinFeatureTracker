import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

path = '/run/media/usr/HDD/preprocessing/3dplots/'

for filename in os.listdir(path):
    if filename.endswith(".png"):
        img = cv2.imread(filename, 0)  # Reads the supplied image in grayscale
        img = cv2.resize(img, (100, 100))

        xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)

        cv2.imwrite(f'before_3d{filename}', img)
        plt.savefig(f'3d_{filename}')
        continue
    else:
        continue
