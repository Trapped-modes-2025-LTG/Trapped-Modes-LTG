import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from scipy.fft import fft2, fftshift, ifft2
from skimage.restoration import unwrap_phase
from matplotlib.colors import LogNorm
import pyfcd.fourier_space as fs
from pyfcd.carriers import Carrier
from pyfcd.height_map import HeightMap
import os
def load_image(path):
    return io.imread(path, as_gray=True).astype(np.float32)

def binarize_image(image):
    """Binarize an image using Otsu's thresholding method."""
    threshold = filters.threshold_otsu(image)
    return image > threshold

def compute_carriers(reference_path, calibration_factor, square_size):
    """
    Compute the carriers for the reference image.

    Parameters:
        reference_path (str): Path to the reference image.
        calibration_factor (float): Calibration factor.
        square_size (float): Size of the square pattern in meters.

    Returns:
        tuple: (reference image, carriers list, detected peaks)
    """
    reference = load_image(reference_path)
    peaks = fs.find_peaks(reference)
    calibration_factor = compute_calibration_factor(peaks, square_size, reference)
    peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
    carriers = [Carrier(reference, calibration_factor, peak, peak_radius) for peak in peaks]
    return reference, carriers, peaks, calibration_factor

def compute_calibration_factor(peaks, square_size, reference):
    """
    Compute the calibration factor using the detected peaks.

    Parameters:
        peaks (array): Detected peak positions.
        square_size (float): Size of the square pattern in meters.
        reference (array): Reference image.

    Returns:
        float: Computed calibration factor.
    """
    pixel_frequencies = fs.pixel_to_wavenumber(reference.shape, peaks)
    pixel_wavelength = 2 * np.pi / np.mean(np.abs(pixel_frequencies))
    physical_wavelength = 2 * square_size
    return physical_wavelength / pixel_wavelength

def compute_phases(displaced_fft, carriers, unwrap=True):
    """
    Compute the phase maps from the displaced image.

    Parameters:
        displaced_fft (array): Fourier transform of the displaced image.
        carriers (list): List of Carrier objects.
        unwrap (bool): Whether to apply phase unwrapping.

    Returns:
        array: Phase maps for each carrier.
    """
    phases = np.zeros((2, *displaced_fft.shape))
    for i, carrier in enumerate(carriers):
        phase_angles = -np.angle(ifft2(displaced_fft * carrier.mask) * carrier.ccsgn)
        phases[i] = unwrap_phase(phase_angles) if unwrap else phase_angles
    return phases

def compute_displacement_field(phases, carriers):
    """
    Compute the displacement field (u, v) from the phase maps.

    Parameters:
        phases (array): Phase maps.
        carriers (list): List of Carrier objects.

    Returns:
        array: Displacement field (u, v).
    """
    det_a = carriers[0].frequencies[1] * carriers[1].frequencies[0] - \
            carriers[0].frequencies[0] * carriers[1].frequencies[1]
    u = (carriers[1].frequencies[0] * phases[0] - carriers[0].frequencies[0] * phases[1]) / det_a
    v = (carriers[0].frequencies[1] * phases[1] - carriers[1].frequencies[1] * phases[0]) / det_a
    return np.array([u, v])
    
def compute_height_map(reference_path, displaced_path, square_size, height=1.0, unwrap=True):
    """
    Compute the height map from two images.

    Parameters:
        reference_path (str): Path to the reference image.
        displaced_path (str): Path to the displaced image.
        square_size (float): Size of the square pattern in meters.
        height (float): Known reference height for calibration.
        unwrap (bool): Whether to apply phase unwrapping.

    Returns:
        HeightMap: Object containing the computed height map.
    """
    reference, carriers, peaks, calibration_factor = compute_carriers(reference_path, None, square_size)

    displaced = load_image(displaced_path)
    displaced_fft = fft2(displaced)

    phases = compute_phases(displaced_fft, carriers, unwrap)
    displacement_field = compute_displacement_field(phases, carriers)
    height_gradient = -displacement_field / height

    height_map = fs.integrate_in_fourier(*height_gradient, calibration_factor)
    return HeightMap(height_map, phases, calibration_factor)





    

    
