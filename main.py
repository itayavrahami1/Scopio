import numpy as np
import scipy.signal
import cv2 as cv

from inventory.Scripts.Service.Image import image_service

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dx_pixel = []
    dz = []

    for i in range(10):
        img1, img2 = image_service.load_images(i)

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        self_corr_img = scipy.signal.fftconvolve(gray1, gray1[::-1,::-1], mode='same')
        corr_img = scipy.signal.fftconvolve(gray1, gray2[::-1,::-1], mode='same')

        max_intensity = (np.unravel_index(np.argmax(self_corr_img), self_corr_img.shape))
        dx_pixel = max_intensity[1] - (np.unravel_index(np.argmax(corr_img), corr_img.shape)[1])

        dz_um = dx_pixel*0.24/(2*np.tan(0.34))
        dz.append(dz_um)

    print(dz)
