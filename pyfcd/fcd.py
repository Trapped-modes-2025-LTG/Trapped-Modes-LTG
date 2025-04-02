import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.fft import fft2, ifft2
from skimage.restoration import unwrap_phase
import pyfcd.fourier_space as fs
from pyfcd.carriers import Carrier

def compute_carriers(reference, calibration_factor, square_size):
    """
    Compute the carriers for the reference image.

    Parameters:
        reference_path (str): Path to the reference image.
        calibration_factor (float): Calibration factor.
        square_size (float): Size of the square pattern in meters.

    Returns:
        tuple: (reference image, carriers list, detected peaks)
    """
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

def height_from_layers(layers):  # TODO: No sé si hay vidrio por ejemplo entre el agua y la cámara cómo cambia esto.
    fluid = layers[-2][1]   #son los índices
    before_camera = layers[-1][1]
    alpha = 1 - before_camera / fluid

    height = 0
    for i in range(len(layers)-1):
        height += effective_height(layers,i)

    return alpha * height

def effective_height(layers,i):
    return layers[2][1] *((layers[i][0]) / (layers[i][1]))

def compute_height_map(reference, displaced, square_size,layers= None, height=None, unwrap=True): #dejo height por las simulaciones
    
    if height is not None:
        if layers is None:
            height = height  # altura efectiva ya.
        else:
            raise Warning("Provide either height or layers, not both.")
    else:
        if layers is None:
            height = 1
        else:
            height = height_from_layers(layers)
    
    reference, carriers, peaks, calibration_factor = compute_carriers(reference, None, square_size)
    displaced_fft = fft2(displaced)
    phases = compute_phases(displaced_fft, carriers, unwrap)
    displacement_field = compute_displacement_field(phases, carriers)
    height_gradient = -displacement_field / height

    height_map = fs.integrate_in_fourier(*height_gradient, calibration_factor)
    return height_map, phases, calibration_factor

    height_gradient = -displacement_field / height
    height_map = fs.integrate_in_fourier(*height_gradient, calibration_factor)
    return height_map, phases, calibration_factor

