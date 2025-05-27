import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.fft import fft2, ifft2, ifftshift
from pyfcd.fourier import fourier
from skimage.draw import disk
import numpy as np

class Carrier:
    def __init__(self, reference_image, calibration_factor, peak, peak_radius):
        self.pixels = peak
        self.frequencies = fourier.pixel_to_wavenumber(reference_image.shape, peak, calibration_factor)
        self.radius = peak_radius
        self.mask = self.peak_mask(reference_image.shape, peak, peak_radius)
        self.ccsgn = self._ccsgn(reference_image)

    def peak_mask(self, shape, pos, r):
        result = np.zeros(shape, dtype=bool)
        result[disk(pos, r, shape=shape)] = True
        return ifftshift(result)

    def _ccsgn(self, reference_image):
        reference_image_fft = fft2(reference_image)
        return np.conj(ifft2(reference_image_fft * self.mask))
