from scipy.fft import fft2, fftshift, fftfreq, ifft2
from skimage.measure import regionprops, label
import numpy as np

def wavenumber(size, calibration_factor=1, shifted=False):
    """
    Compute the wavenumber (spatial frequency) vector.
    
    Parameters:
        size (int): Length of the frequency vector
        calibration_factor (float): Scaling factor for frequency units (default=1)
        shifted (bool): Whether to center the frequencies (fftshift) (default=False)
    
    Returns:
        np.ndarray: 1D array of wavenumbers
    """
    frequencies = fftfreq(size, calibration_factor/(2*np.pi))
    return fftshift(frequencies) if shifted else frequencies


def wavenumber_meshgrid(shape, calibration_factor=1, shifted=False):
    """
    Create a 2D meshgrid of wavenumbers.
    
    Parameters:
        shape (tuple): Grid dimensions (rows, cols)
        calibration_factor (float): Scaling factor for frequency units (default=1)
        shifted (bool): Whether to center the frequencies (default=False)
    
    Returns:
        tuple: (kx, ky) meshgrid arrays in 'ij' indexing
    """
    k_rows = wavenumber(shape[0], calibration_factor, shifted)
    k_cols = wavenumber(shape[1], calibration_factor, shifted)
    return np.meshgrid(k_rows, k_cols, indexing='ij')


def remove_degeneracy(kx, ky, shape):
    """
    Remove Fourier space degeneracy at Nyquist frequencies.
    
    Parameters:
        kx (np.ndarray): Horizontal wavenumbers mesh
        ky (np.ndarray): Vertical wavenumbers mesh
        shape (tuple): Original array shape (rows, cols)
    
    Returns:
        None: Modifies input arrays in-place
    """
    if shape[1] % 2 == 0:
        kx[:, shape[1]//2+1] = 0  # Remove degeneracy at kx=Nx/2 leading to imaginary part.

    if shape[0] % 2 == 0:
        ky[shape[0]//2+1, :] = 0  # Remove degeneracy at ky=Ny/2 leading to imaginary part.


def pixel_to_wavenumber(image_shape, locations, calibration_factor=1):
    """
    Convert pixel coordinates to wavenumber values.
    
    Parameters:
        image_shape (tuple): Shape of the original image (rows, cols)
        locations (np.ndarray|tuple): Pixel coordinate(s) to convert
        calibration_factor (float): Scaling factor (default=1)
    
    Returns:
        np.ndarray: Corresponding wavenumber coordinates
    """
    k_space_rows = wavenumber(image_shape[0], calibration_factor, shifted=True)
    k_space_cols = wavenumber(image_shape[1], calibration_factor, shifted=True)

    if isinstance(locations[0], np.ndarray):
        return np.array([[k_space_rows[loc[0]], k_space_cols[loc[1]]] for loc in locations])
    else:
        return np.array([k_space_rows[locations[0]], k_space_cols[locations[1]]])


def integrate_in_fourier(gradient_x, gradient_y, calibration_factor=1):  # TODO: Agregar contribución lineal opcional.
    """
    Reconstruct a field from its gradients using Fourier integration.
    
    Parameters:
        gradient_x (np.ndarray): x-component of gradient field
        gradient_y (np.ndarray): y-component of gradient field
        calibration_factor (float): Scaling factor (default=1)
    
    Returns:
        np.ndarray: Reconstructed scalar field
    """
    x_mean = np.mean(gradient_x)
    y_mean = np.mean(gradient_y)
    kx, ky = wavenumber_meshgrid(gradient_x.shape, calibration_factor)  # TODO: Precalcular esto en FCD() con referencia.
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1

    remove_degeneracy(kx, ky, gradient_x.shape)

    gradient_x_hat, gradient_y_hat = fft2(gradient_x), fft2(gradient_y)
    integrated_hat = (-1.0j * kx * gradient_x_hat + -1.0j * ky * gradient_y_hat) / k2
    
    f = np.real(ifft2(integrated_hat))
    nx, ny = gradient_x.shape
    X,Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    f = f+X*x_mean+Y*y_mean
    return f


def find_peak_locations(image, threshold, no_peaks):
    """
    Detect brightest peaks in thresholded image.
    
    Parameters:
        image (np.ndarray): Input image
        threshold (float): Intensity threshold for peak detection
        no_peaks (int): Maximum number of peaks to return
    
    Returns:
        list: Coordinates of detected peaks [(row1, col1), ...]
    """
    blob_image = np.array(image > threshold)  # TODO: el np.array() no es necesario pero sino piensa que blob es tipo bool.

    # make the borders false
    blob_image[0] *= False
    blob_image[-1] *= False
    blob_image[..., 0] *= False
    blob_image[..., -1] *= False

    blob_data = regionprops(label(blob_image.astype(np.uint8)))

    def blob_max_pixel_intensity(blob):
        pixels_with_coords = [(image[tuple(c)], c) for c in blob.coords]
        return max(pixels_with_coords, key=lambda x: x[0])

    blobs_with_max_intensity_and_coord = [blob_max_pixel_intensity(blob) for blob in blob_data]
    sorted_blobs = sorted(blobs_with_max_intensity_and_coord, key=lambda x: x[0])
    return [peak[1] for peak in sorted_blobs[:no_peaks]] 


def find_peaks(image):
    """
    Identify dominant frequency peaks in FFT spectrum.
    
    Parameters:
        image (np.ndarray): Input image
    
    Returns:
        tuple: Coordinates of (rightmost_peak, perpendicular_peak)
    """
    image_fft = fftshift(np.abs(fft2(image - np.mean(image))))

    def highpass_mask():
        ks_mesh_x, ks_mesh_y = wavenumber_meshgrid(image_fft.shape, shifted=True)
        kmin = 4 * np.pi / min(image.shape)
        return (ks_mesh_x**2 + ks_mesh_y**2) > kmin**2

    def angles(testing_peak):
        testing_peak_frequency = pixel_to_wavenumber(image_fft.shape, testing_peak)
        return abs(np.arctan2(*testing_peak_frequency))

    def dependendancy(testing_peak):
        first_peak_frequency = pixel_to_wavenumber(image_fft.shape, rightmost_peak)
        testing_peak_frequency = pixel_to_wavenumber(image_fft.shape, testing_peak)
        return abs(np.dot(first_peak_frequency, testing_peak_frequency))

    image_fft *= highpass_mask()
    threshold = 0.5 * np.max(image_fft)

    peak_locations = find_peak_locations(image_fft, threshold, 4)
    rightmost_peak = min(peak_locations, key=angles)
    perpendicular_peak = min(peak_locations, key=dependendancy)

    return rightmost_peak, perpendicular_peak

    peak_locations = find_peak_locations(image_fft, threshold, 4)
    rightmost_peak = min(peak_locations, key=angles)
    perpendicular_peak = min(peak_locations, key=dependendancy)

    return rightmost_peak, perpendicular_peak
