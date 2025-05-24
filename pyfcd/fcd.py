import numpy as np
from pyfcd.carriers import Carrier
from scipy.fft import fft2, fftshift, fftfreq, ifft2
from skimage.restoration import unwrap_phase
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

class fcd:
    
    @classmethod
    def compute_height_map(cls,reference, displaced, square_size,layers= None, height=None, unwrap=True): #dejo height por las simulaciones
        
        if height is not None:
            if layers is None:
                height = height  # altura efectiva ya.
            else:
                raise Warning("Provide either height or layers, not both.")
        else:
            if layers is None:
                height = 1
            else:
                height = cls.height_from_layers(layers)
        
        carriers, calibration_factor = cls.compute_carriers(reference, square_size)
        displaced_fft = fft2(displaced)
        phases = cls.compute_phases(displaced_fft, carriers, unwrap)
        displacement_field = cls.compute_displacement_field(phases, carriers)
        
        height_gradient = -displacement_field / height
        height_map = fourier.integrate_in_fourier(*height_gradient, calibration_factor)
        
        return height_map, phases, calibration_factor
    
    @classmethod
    def height_from_layers(cls,layers):  # TODO: No sé si hay vidrio por ejemplo entre el agua y la cámara cómo cambia esto.
        fluid = layers[-2][1]   #son los índices
        before_camera = layers[-1][1]
        alpha = 1 - before_camera / fluid
    
        height = 0
        for i in range(len(layers)-1):
            height += cls.effective_height(layers,i)
    
        return alpha * height
    
    @classmethod
    def effective_height(cls, layers,i):
        return layers[2][1] *((layers[i][0]) / (layers[i][1]))
    
    @classmethod
    def compute_carriers(cls,reference, square_size):
        """
        Compute the carriers for the reference image.
    
        Parameters:
            reference_path (str): Path to the reference image.
            calibration_factor (float): Calibration factor.
            square_size (float): Size of the square pattern in meters.
    
        Returns:
            tuple: (reference image, carriers list, detected peaks)
        """
        
        calibration_factor, peaks = cls.compute_calibration_factor(square_size, reference)
        peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
        carriers = [Carrier(reference, calibration_factor, peak, peak_radius) for peak in peaks]
        return  carriers, calibration_factor
    
    @classmethod
    def compute_calibration_factor(square_size, reference, plot = False):
        """
        Compute the calibration factor using the detected peaks.
    
        Parameters:
            peaks (array): Detected peak positions.
            square_size (float): Size of the square pattern in meters.
            reference (array): Reference image.
    
        Returns:
            float: Computed calibration factor.
        """
        peaks = fourier.find_peaks(reference)
        pixel_frequencies = fourier.pixel_to_wavenumber(reference.shape, peaks)
        pixel_wavelength = 2 * np.pi / np.mean(np.abs(pixel_frequencies))
        physical_wavelength = 2 * square_size
        
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(reference, cmap='gray')
            ax.set_title(f"Calibration factor: {physical_wavelength / pixel_wavelength:.2f} dist/px")
            ax.plot([200,200+pixel_wavelength], [201,201], '.-', label = r'$\lambda$')
            ax.set_xlabel("X (pix)")
            ax.set_ylabel("Y (pix)")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        return physical_wavelength / pixel_wavelength, peaks
    
    
    @classmethod
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
    
    @classmethod
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
    
    
class fourier:
    
    @classmethod
    def find_peaks(cls, image):
        """
        Identify dominant frequency peaks in FFT spectrum.
        
        Parameters:
            image (np.ndarray): Input image
        
        Returns:
            tuple: Coordinates of (rightmost_peak, perpendicular_peak)
        """
        image_fft = fftshift(np.abs(fft2(image - np.mean(image))))
    
        def highpass_mask():
            ks_mesh_x, ks_mesh_y = cls.wavenumber_meshgrid(image_fft.shape, shifted=True)
            kmin = 4 * np.pi / min(image.shape)
            return (ks_mesh_x**2 + ks_mesh_y**2) > kmin**2
    
        def angles(testing_peak):
            testing_peak_frequency = cls.pixel_to_wavenumber(image_fft.shape, testing_peak)
            return abs(np.arctan2(*testing_peak_frequency))
    
        def dependendancy(testing_peak):
            first_peak_frequency = cls.pixel_to_wavenumber(image_fft.shape, rightmost_peak)
            testing_peak_frequency = cls.pixel_to_wavenumber(image_fft.shape, testing_peak)
            return abs(np.dot(first_peak_frequency, testing_peak_frequency))
    
        image_fft *= highpass_mask()
        threshold = 0.5 * np.max(image_fft)
    
        peak_locations = cls.find_peak_locations(image_fft, threshold, 4)
        rightmost_peak = min(peak_locations, key=angles)
        perpendicular_peak = min(peak_locations, key=dependendancy)
    
        return rightmost_peak, perpendicular_peak
    
    @classmethod
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
    
    @classmethod
    def wavenumber_meshgrid(cls, shape, calibration_factor=1, shifted=False):
        """
        Create a 2D meshgrid of wavenumbers.
        
        Parameters:
            shape (tuple): Grid dimensions (rows, cols)
            calibration_factor (float): Scaling factor for frequency units (default=1)
            shifted (bool): Whether to center the frequencies (default=False)
        
        Returns:
            tuple: (kx, ky) meshgrid arrays in 'ij' indexing
        """
        k_rows = cls.wavenumber(shape[0], calibration_factor, shifted)
        k_cols = cls.wavenumber(shape[1], calibration_factor, shifted)
        return np.meshgrid(k_rows, k_cols, indexing='ij')
    
    @classmethod
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
    
    @classmethod
    def pixel_to_wavenumber(cls,image_shape, locations, calibration_factor=1):
        """
        Convert pixel coordinates to wavenumber values.
        
        Parameters:
            image_shape (tuple): Shape of the original image (rows, cols)
            locations (np.ndarray|tuple): Pixel coordinate(s) to convert
            calibration_factor (float): Scaling factor (default=1)
        
        Returns:
            np.ndarray: Corresponding wavenumber coordinates
        """
        k_space_rows = cls.wavenumber(image_shape[0], calibration_factor, shifted=True)
        k_space_cols = cls.wavenumber(image_shape[1], calibration_factor, shifted=True)
    
        if isinstance(locations[0], np.ndarray):
            return np.array([[k_space_rows[loc[0]], k_space_cols[loc[1]]] for loc in locations])
        else:
            return np.array([k_space_rows[locations[0]], k_space_cols[locations[1]]])
    
    @classmethod
    def integrate_in_fourier(cls,gradient_x, gradient_y, calibration_factor=1):  # TODO: Agregar contribución lineal opcional.
        """
        Reconstruct a field from its gradients using Fourier integration.
        
        Parameters:
            gradient_x (np.ndarray): x-component of gradient field
            gradient_y (np.ndarray): y-component of gradient field
            calibration_factor (float): Scaling factor (default=1)
        
        Returns:
            np.ndarray: Reconstructed scalar field
        """    
        ky, kx = cls.wavenumber_meshgrid(gradient_x.shape, calibration_factor)  # TODO: Precalcular esto en FCD() con referencia.
        k2 = kx ** 2 + ky ** 2
        k2[0, 0] = 1
    
        cls.remove_degeneracy(kx, ky, gradient_x.shape)
    
        gradient_x_hat, gradient_y_hat = fft2(gradient_x), fft2(gradient_y)
        integrated_hat = (-1.0j * kx * gradient_x_hat + -1.0j * ky * gradient_y_hat) / k2
    
        return np.real(ifft2(integrated_hat))
    
    @classmethod
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
    
    