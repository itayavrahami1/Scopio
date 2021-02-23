import numpy as np
import cv2 as cv
import os


def load_images(img_path, files):

    im1_path = os.path.join(img_path, files[0])
    im2_path = os.path.join(img_path, files[1])

    img1 = cv.imread(im1_path)
    img2 = cv.imread(im2_path)

    return img1, img2
