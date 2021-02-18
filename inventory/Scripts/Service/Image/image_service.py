import numpy as np
import cv2 as cv


def load_images(num):

    img1 = cv.imread(f'Interview/Interview/{num}/im1.1028x687.jpg')
    img2 = cv.imread(f'Interview/Interview/{num}/im2.1028x687.jpg')

    return img1, img2
