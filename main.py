import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import cv2 as cv
import os

from inventory.Scripts.Service.Image import image_service

if __name__ == '__main__':

    path = r'C:\Users\itay.a\Coding\OpenCV\Interview\Interview'
    dx_pixel = []
    dz = []
    labels = []

    for curr_dir in os.scandir(path):

        try:
            label = int(curr_dir.name)
            labels.append(label)
            files = os.listdir(os.path.join(path,curr_dir))
            img1, img2 = image_service.load_images(os.path.join(path,curr_dir), files)

            gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

            self_corr_img = scipy.signal.fftconvolve(gray1, gray1[::-1, ::-1], mode='same')
            corr_img = scipy.signal.fftconvolve(gray1, gray2[::-1,::-1], mode='same')

            max_intensity = (np.unravel_index(np.argmax(self_corr_img), self_corr_img.shape))
            dx_pixel = max_intensity[1] - (np.unravel_index(np.argmax(corr_img), corr_img.shape)[1])

            dz_um = dx_pixel*0.24/(2*np.tan(0.34))
            dz.append(dz_um)
        except:
            continue

    plt.plot(labels, dz, marker='*')
    plt.title('dz [um] Vs. image label')
    plt.grid()
    plt.xlabel('Label')
    plt.ylabel('dz [um]')
    plt.show()
