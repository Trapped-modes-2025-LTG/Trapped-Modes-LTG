import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from pyfcd.fourier import fourier
from pyfcd.carriers import Carrier
    
class fcd:
    def __init__(self):
        self.fft = fourier()
        self.carrier = Carrier()
        
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
    def compute_calibration_factor(cls,square_size, reference, plot = False):
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
            ax.set_title(f"Calibration factor: \n {physical_wavelength / pixel_wavelength} dist/px")
            ax.plot([np.shape(reference)[0]/2,np.shape(reference)[0]/2+pixel_wavelength], [np.shape(reference)[1]/2,np.shape(reference)[1]/2], '.-', label = r'$\lambda$')
            ax.set_xlabel("X (pix)")
            ax.set_ylabel("Y (pix)")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        return physical_wavelength / pixel_wavelength, peaks
    
    @classmethod
    def compute_phases(cls,displaced_fft, carriers, unwrap=True):
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
    def compute_displacement_field(cls,phases, carriers):
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
    
    
    @staticmethod
    def fft_peaks(image):
        """
        Plot the FFT spectrum and highlight the detected peaks and selected directions.
        
        Parameters:
            image (np.ndarray): Input grayscale image.
        """
        image_fft_raw = fftshift(np.abs(fft2(image - np.mean(image))))
        log_fft = np.log1p(image_fft_raw)
        
        ks_mesh_x, ks_mesh_y = fourier.wavenumber_meshgrid(image_fft_raw.shape, shifted=True)
        kmin = 4 * np.pi / min(image.shape)
        mask = (ks_mesh_x**2 + ks_mesh_y**2) > kmin**2
        image_fft = image_fft_raw * mask

        threshold = 0.5 * np.max(image_fft)
        peak_locations = fourier.find_peak_locations(image_fft, threshold, 4)
        rightmost_peak, perpendicular_peak = fourier.find_peaks(image)

        fig, ax = plt.subplots()
        ax.imshow(log_fft, origin='lower', cmap='magma')
        ax.set_title("FFT Spectrum with Peaks")
        ax.set_xlabel(r"$k_x$ (1/pix)")
        ax.set_ylabel(r"$k_y$ (1/pix)")

        for i, (y, x) in enumerate(peak_locations):
            ax.plot(x, y, 'k.', markersize=6)
            ax.text(x+5, y+5, f"peak {i+1}", color='white', fontsize=9)

        ax.plot(rightmost_peak[1], rightmost_peak[0], 'r.', label='Rightmost peak')
        ax.plot(perpendicular_peak[1], perpendicular_peak[0], 'b.', label=r'$\perp$ peak')

        ax.legend()
        plt.tight_layout()
        plt.show()
        

    
    