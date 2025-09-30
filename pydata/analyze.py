'''
This script contains several functions to analyze measured data.
All of them are included in a single class called `analyze` for organizational purposes.
Each function is explained when it is called.
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyfcd.fcd import fcd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import uniform_filter

from skimage import io
from skimage.measure import regionprops, label, find_contours
from scipy.signal import find_peaks
from skimage.transform import warp, rotate

from tqdm import tqdm

class analyze:
    @classmethod
    def load_image(cls,path):
        '''
        Loads a grayscale image and converts it to float32.
    
        Parameters
        ----------
        path: str
            Path to the image file.
    
        Returns
        -------
        ndarray
            2D grayscale image as a float32 NumPy array.
        '''
        return io.imread(path, as_gray=True).astype(np.float32)
    
    @classmethod
    def mask(cls,image, smoothed = 14, show_mask = False, find_center = False):
        '''
        Creates a mask for the region related to the floating structure and detects its center.
    
        Parameters
        ----------
        image : ndarray
            2D grayscale image as a NumPy array .
        smoothed : int, optional
            Size of the smoothing filter applied before masking.
        show_mask : bool, default=False
            If True, displays the generated mask.
        center: bool, default=False
            If True, calculates the center as a tuple (cy, cx)
    
        Returns
        -------
        mask : ndarray of bool
            2D boolean mask of the region of interest. Pixels inside the region are True.
        center : tuple of int, optional
            (cy, cx) coordinates of the region's centroid. Returned only if `center=True`.

        '''

        smooth = uniform_filter(image, size=smoothed)
        threshold = np.mean(smooth)
        Mask = smooth < threshold
        labels = label(Mask)  
        regions = regionprops(labels)
        regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)

        r = regions_sorted[0]

        mask = labels == r.label  
        masked = image * mask
        
        if show_mask: 
            fig, ax = plt.subplots(1, 3, figsize=(10,4))
                
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Original image")

            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("Mask")

            ax[2].imshow(masked, cmap="gray")
            ax[2].set_title("Masked image")

            for a in ax:
                a.axis("off")  # opcional, quita los ejes

            plt.tight_layout()
            plt.show()

        if find_center: 
            return mask, cls.center(mask)
        else:
            return mask
        
        
    @classmethod
    def center(cls, mask):

        '''
        Calculates the center of the region inside the floating structure.
    
        Parameters
        ----------
        mask : ndarray of bool
            2D boolean mask of the region of structures cavity.
    
        Returns
        -------
        center : tuple of int
            (cy, cx) coordinates of the region's centroid.
        '''

        inv_mask = ~mask

        label_img = label(inv_mask)
        props = regionprops(label_img)

        n_rows, n_cols = mask.shape
        hole_regions = []

        for region in props:        # TODO: pasar bbox a funcion aparte _
            min_row, min_col, max_row, max_col = region.bbox
            if min_row > 0 and min_col > 0 and max_row < n_rows and max_col < n_cols: 
                hole_regions.append(region)                                           
                                                                         
        if hole_regions:
            largest_hole = max(hole_regions, key=lambda r: r.area)
            Cy, Cx = largest_hole.centroid
            cy, cx = int(Cy), int(Cx)
            
            center = (cy, cx)

        return center
    
    @classmethod
    def folder(cls, reference_path, displaced_dir, layers, square_size,
               smoothed=None, polar = False,show_mask=False, **kwargs):
        '''
        Processes a folder of ".tif" images to compute height maps using the FCD method.
    
        Parameters
        ----------
        reference_path : str
            Path to the reference image used by the FCD method.
        displaced_dir : str
            Path to the folder containing the displaced images.
        layers : list
            List of layer parameters for pyfcd (check pyfcd docs).
        square_size : float
            Size of square pattern at rest.
        smoothed : int, optional
            Size of the smoothing filter applied before masking.
        show_mask : bool, default=False
            If True, displays the generated mask and contours.
    
        Returns
        -------
        None
            Saves height maps in a "maps" folder inside displaced_dir.
        '''
    
        reference = cls.load_image(reference_path)
        output_dir = os.path.join(displaced_dir, 'maps')
        os.makedirs(output_dir, exist_ok=True)
    
        calibration_saved = False
        file_list = sorted(os.listdir(displaced_dir))
        tif_list = [f for f in file_list if f.endswith('.tif') and 'reference' not in f]
    
        existing_maps = sorted(f for f in os.listdir(output_dir) if f.endswith('_map.npy'))
        start_index = len(existing_maps)
    
        centers_path = os.path.join(displaced_dir, 'centers.txt')
        angles_path = os.path.join(displaced_dir, 'angles.txt')
        
        if polar: 
            if not os.path.exists(angles_path):
                open(centers_path, "w").close()
            # also resume from centers.txt line count
            with open(angles_path, "r") as f:
                lines = f.readlines()
            start_index = max(start_index, len(lines))
            
        if smoothed:
            if not os.path.exists(centers_path):
                open(centers_path, "w").close()
            # also resume from centers.txt line count
            with open(centers_path, "r") as f:
                lines = f.readlines()
            start_index = max(start_index, len(lines))
    
            if show_mask:
                displaced_path = os.path.join(displaced_dir, tif_list[min(10, len(tif_list)-1)])
                displaced_image = cls.load_image(displaced_path)
    
                while True:
                    mask = cls.mask(displaced_image, smoothed=smoothed, show_mask=True)
                    plt.pause(10)
                    plt.close("all")
    
                    message = input("Continue with this mask? [Y,n]: ")
    
                    if message == "Y":
                        break
                    elif message == "n":
                        try:
                            new_val = input("Enter new smoothed value (int): ")
                            smoothed = int(new_val)
                        except ValueError:
                            print("Invalid input, keeping previous smoothed.")
            else:
                if show_mask:
                    raise(ValueError("If show_mask == True, expect smoothed too"))
    
        # Main loop
        for i, fname in tqdm(enumerate(tif_list[start_index:], start=start_index)):
            displaced_path = os.path.join(displaced_dir, fname)
            displaced_image = cls.load_image(displaced_path)
    
            mask_applied = False
            
            if smoothed:
                mask = cls.mask(displaced_image, smoothed=smoothed)
                image_to_use = np.where(mask == 1, reference, displaced_image)
    
                center = cls.center(mask)
                with open(centers_path, "a") as f:  # append
                    f.write(f"{i}\t{center}\n")
    
                mask_applied = True
                
                if polar:
                    import cv2
                    contour = cls._set_contour(mask) 
                    contour = np.array(contour, dtype=np.float32)
                    contour_cv = contour[:, ::-1]  # (y,x) -> (x,y) 
                    _, _, angle = cv2.fitEllipse(contour_cv)
                    
                    with open(angles_path, "a") as f:  # append
                        f.write(f"{i}\t{angle}\n")
                        
            else:
                image_to_use = displaced_image
    
            height_map, _, calibration_factor = fcd.compute_height_map(
                reference,
                image_to_use,
                square_size,
                layers
            )
    
            if mask_applied:
                height_map *= ~mask
    
            base_name = fname.replace('.tif', '')
            
            if polar is False:
                output_path = os.path.join(output_dir, f"{base_name}_map.npy")
                np.save(output_path, height_map)
            else:
                output_path = os.path.join(output_dir, f"{base_name}_map_polar.npy")
                height_map_polar = cls.polar(img = height_map, angle = angle,**kwargs)
                output_path = os.path.join(height_map_polar, f"{base_name}_map_polar.npy")
    
            if not calibration_saved:
                calibration_path = os.path.join(output_dir, 'calibration_factor.npy')
                np.save(calibration_path, np.array([calibration_factor]))
                calibration_saved = True

    @staticmethod
    def video(maps_dir, calibration_factor=None, frame_final=300, n_extra_frames=20):

        '''
        Creates and saves a video from a series of height maps.

        Parameters
        ----------
        map_dir : str
            Path to the folder containing '_map.npy' files and calibration factor.
        calibration_factor : float, optional, default=None
            Scaling factor to convert pixel units to meters. 
        frame_final : int, optional, default=300
            Maximum frame index to display. If the number of maps is smaller, the last available frame is used..
        n_extra_frames : TYPE, optional, default=20
            Number of extra frames to hold the last frame at the end of the animation.
            

        Returns
        -------
        None
            Saves a .mp4 video in the same directory as the script.

        '''
        
        if calibration_factor is None:
            calibration_path = os.path.join(maps_dir, 'calibration_factor.npy')
            if not os.path.isfile(calibration_path):
                raise FileNotFoundError(f"calibration_factor.npy was not found in {maps_dir}")
            calibration_factor = np.load(calibration_path).item()
        elif not isinstance(calibration_factor, (float, int)):
            raise TypeError("calibration_factor must be an integer or float")
    
        file_list = sorted([f for f in os.listdir(maps_dir) if f.endswith('_map.npy')])
        df = pd.DataFrame({'filename': file_list})
        df['full_path'] = df['filename'].apply(lambda x: os.path.join(maps_dir, x))
        df['data'] = df['full_path'].apply(np.load)
    
        min_val = min(np.min(d) for d in df['data'])
        max_val = max(np.max(d) for d in df['data'])
    
        height_map = df['data'].iloc[0]
        n_rows, n_cols = height_map.shape
        x = np.linspace(0, n_cols, n_cols) * calibration_factor
        y = np.linspace(0, n_rows, n_rows) * calibration_factor
        x_mesh, y_mesh = np.meshgrid(x, y)
    
        fig, ax = plt.subplots()
        im = ax.contourf(x_mesh * 1e3, y_mesh * 1e3, height_map * 1e3, 100, cmap='magma',
                         vmin=min_val * 1e3, vmax=max_val * 1e3)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Height [mm]', rotation=270, labelpad=15)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_aspect("equal")
    
        def update(i):
            if i >= len(df):
                i = min(frame_final, len(df) - 1)
            ax.clear()
            cont = ax.contourf(x_mesh * 1e3, y_mesh * 1e3, df['data'].iloc[i] * 1e3, 100, cmap='magma',
                               vmin=min_val * 1e3, vmax=max_val * 1e3)
            ax.set_title(f'Frame {i}')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_aspect("equal")
            return cont.collections
    
        total_frames = len(df) + n_extra_frames
        ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False, interval=100)
    
        base_dir = os.path.dirname(__file__)
        output_path = os.path.join(base_dir, 'video.mp4')
        ani.save(output_path, writer='ffmpeg', fps=30)
        print(f"Saved in: {output_path}")

    @classmethod
    def block_split(cls,map_folder, t_limit=None, num_blocks=64, block_index=0):
        '''
        Splits a set of height maps into spatial blocks and returns the temporal stack of one block.
        
        Parameters
        ----------
        map_folder : str
            Path to the folder containing '_map.npy' files.
        t_limit : int, default=None
            Sets a limit to the amount files to analize.
        num_blocks : int, optional, default=64
            Total number of blocks in which the map is divided.
        block_index : int, optional, default=0
            Index of the block to process.

        Returns
        -------
        maps : ndarray of shape (block_size, block_size, N)
            3D array containing the selected block over N time frames.
            
        Pixels that were zero in the first map are set to NaN.

        '''

        file_list = sorted([
            f for f in os.listdir(map_folder)
            if f.endswith('_map.npy') and 'calibration_factor' not in f
        ])
        file_list = file_list[:t_limit]
    
        initial_map = np.load(os.path.join(map_folder, file_list[0]))
        H, W = initial_map.shape
    
        mask_ceros = (initial_map == 0)
        mask_validos = ~mask_ceros
    
        blocks_per_row = int(np.sqrt(num_blocks))
        block_size = H // blocks_per_row
    
        i = block_index // blocks_per_row
        j = block_index % blocks_per_row
    
        maps = []
        for f in file_list:
            m = np.load(os.path.join(map_folder, f))
            block = m[i*block_size : (i+1)*block_size, j*block_size : (j+1)*block_size]
            mask_block = mask_validos[i*block_size : (i+1)*block_size, j*block_size : (j+1)*block_size]
            block_masked = np.where(mask_block, block, np.nan)
            maps.append(block_masked)
    
        maps = np.stack(maps, axis=0)  # (N, block_size, block_size)
        
        return np.transpose(maps, (1, 2, 0))
    
    @classmethod
    def spectrogram(cls,map_folder = None,array = None, fs=125, show = False, **kwargs):
    
        '''

        Computes the spectrogram of a signal or a block of signals.

        Parameters
        ----------
        map_folder : str, optional, default=None
            Path to a folder containing the reference images for block processing.
        array : ndarray, optional, default=None
            1D array representing a single time series for spectrogram calculation.
        fs : int, optional, default=125
            Sampling frequency of the signal in Hz.
        show : bool, optional, default=False
            If True, displays the spectrogram using matplotlib.
        **kwargs : dict
            Additional keyword arguments for customization:
        - Spectrogram parameters (passed to `scipy.signal.spectrogram`):
            - nperseg : int
            - noverlap : int
            - window  : str or tuple or array_like
        - Block processing parameters (for `map_folder`):
            - t_limit      : tuple (start, end)
            - num_blocks   : int
            - block_index  : int

        Returns
        -------
        If `array` is provided and `map_folder` is None:
            t : ndarray
                Array of segment times.
            f : ndarray
                Array of segment frequencies.
            Sxx : ndarray
                Spectrogram of the signal (shape: [frequencies, times]).
            
        If `map_folder` is provided and `array` is None:
            t : ndarray
                Array of segment times.
            f : ndarray
                Array of segment frequencies.
            Sxx_all : ndarray
                Spectrogram of each pixel in the block (shape: [ny, nx, nf, nt]).
            Sxx_avg : ndarray
                Average spectrogram over all pixels (shape: [nf, nt]).
        Notes
        -----
        Processing time:
            - Single image (`map_folder=None`): ~121 ms (show=True) / 344 μs ± 3.78 μs (show=False)
            - Entire dataset (`map_folder`): ~1h 26min 24s
            - Single reference map (~81 s)
        

        '''
    
        signal_kwargs = {k: kwargs[k] for k in ['nperseg', 'noverlap', 'window'] if k in kwargs}
        block_kwargs = {k: kwargs[k] for k in ['t_limit', 'num_blocks', 'block_index'] if k in kwargs}
        from scipy import signal
        if map_folder == None:      
            if array is not None:   # Spectrogram for some point
        
                f, t, Sxx = signal.spectrogram(array, fs=fs, **signal_kwargs)
                if show:
                    plt.figure(figsize=(8, 4))
                    plt.pcolormesh(t, f, np.log10(Sxx), shading='gouraud')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.title('Spectrogram of some point')
                    plt.colorbar(label='Amplitude (mm)')
                    plt.tight_layout()
                    plt.show()
                return t,f,Sxx
        
            else:
                raise ValueError("Any map_folder or array is needed")
        else:
            if array == None:       # Spectrogram for a block
                maps = cls.block_split(map_folder, **block_kwargs)
        
                ny, nx, N = maps.shape
        
                # Example to get output shapes
                f, t, Sxx_example = signal.spectrogram(maps[0, 0, :], fs=fs, **signal_kwargs)
                nf = len(f)
                nt = len(t)
        
                Sxx_all = np.empty((ny, nx, nf, nt))

                # Loop over pixels
                for iy in range(ny):
                    for ix in range(nx):
                        ts = maps[iy, ix, :]
                        if np.isnan(ts).all():
                            Sxx_all[iy, ix] = np.nan
                        else:
                            if np.isnan(ts).any():
                                ts = np.interp(
                                    np.arange(len(ts)),
                                    np.arange(len(ts))[~np.isnan(ts)],
                                    ts[~np.isnan(ts)]
                                    )
                            _, _, Sxx = signal.spectrogram(ts, fs=fs, **signal_kwargs)
                            Sxx_all[iy, ix] = Sxx

                Sxx_avg = np.nanmean(Sxx_all, axis=(0, 1))  # shape (nf, nt)
            
                if show:
                    plt.figure(figsize=(8, 4))
                    plt.pcolormesh(t, f, np.log10(Sxx), shading='gouraud')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.title('Average Spectrogram over block')
                    plt.colorbar(label='ln(Amplitude)')
                    plt.tight_layout()
                    plt.show()
        
                return t,f, Sxx_all, Sxx_avg
            else:
                ValueError("map_folder or array is needed, not both")
     
    @classmethod
    def block_amplitude(cls, map_folder, f0=None, tasa=500, mode=1, num_blocks=64, block_index=0, zero = 0):  
        '''        
        Computes the amplitude and phase of harmonic components for a spatial block of height maps.

        Parameters
        ----------
        map_folder : str
            Path to the folder containing '_map.npy' files.
        f0 : float, optional, default=None
            Fundamental frequency to analyze. If None, it is automatically estimated from the mean spectrum.
        tasa : int, optional, default=500
            Sampling rate of the temporal signal in Hz.
        mode : int, optional, default=1
            Number of harmonics to compute (including the fundamental).
        num_blocks : int, optional, default=64
            Total number of blocks in which the map is divided.
        block_index : int, optional, default=0
            Index of the block to process.

        zero : float, optional, default=0
            Value to subtract from the maps before analysis, for example Mode 0.

        Returns
        -------
        harmonics : list of float
            Frequencies of the fundamental and harmonic components.
        amps : ndarray of shape (block_size, block_size, mode+1)
            Amplitude of each harmonic per pixel in the block.
        phases : ndarray of shape (block_size, block_size, mode+1)
            Phase of each harmonic per pixel in the block.
        mean_spectrum : ndarray
            Mean magnitude spectrum over all valid pixels (used to detect f0 if None).
        fft_freqs : ndarray
            Frequencies corresponding to the FFT output.

        '''

        file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])

        initial_map = np.load(os.path.join(map_folder, file_list[0]))
        H, W = initial_map.shape
        
        mask_ceros = (initial_map == 0)
        
        mask_validos = ~mask_ceros

        blocks_per_row = int(np.sqrt(num_blocks))
        block_size = H // blocks_per_row

        i = block_index // blocks_per_row
        j = block_index % blocks_per_row

        maps = []
        for f in file_list:
            m = np.load(os.path.join(map_folder, f)) - zero

            block = m[i*block_size : (i+1)*block_size,j*block_size : (j+1)*block_size]
            mask_block = mask_validos[i*block_size : (i+1)*block_size,j*block_size : (j+1)*block_size]

            block_masked = np.where(mask_block, block, np.nan)
            maps.append(block_masked)

        maps = np.stack(maps, axis=0)  # (N, block_size, block_size)
        maps = np.transpose(maps, (1, 2, 0))  # (block_size, block_size, N)

        ny, nx, N = maps.shape
        dt = 1 / tasa

        fft_vals = np.fft.fft(maps, axis=-1)
        fft_freqs = np.fft.fftfreq(N, d=dt)

        pos_freqs = fft_freqs >= 0
        fft_vals = fft_vals[:, :, pos_freqs]
        fft_freqs = fft_freqs[pos_freqs]
        

        if f0 is None:
            # Promediar magnitud sobre todos los píxeles para robustez
            mean_spectrum = np.nanmean(np.abs(fft_vals), axis=(0, 1))
            peaks, _ = find_peaks(mean_spectrum)
            if len(peaks) == 0:
                return np.zeros(mode), np.full((ny, nx, mode), None, dtype=object), np.full((ny, nx, mode), None, dtype=object), None, None
            max_peak_index = peaks[np.argmax(mean_spectrum[peaks])]
            f0 = fft_freqs[max_peak_index]  

        harmonics = [f0 * n for n in range(0, mode)]
        indices = [np.argmin(np.abs(fft_freqs - f)) for f in harmonics]

        amps = np.zeros((ny, nx, mode+1))
        phases = np.zeros((ny, nx, mode+1))

        for k, idx in enumerate(indices):
            harmonic_vals = fft_vals[:, :, idx]
            if k == 0:  
                amps[:, :, k] = np.abs(harmonic_vals) / N
            else:
                amps[:, :, k] = 2 * np.abs(harmonic_vals) / N
            phases[:, :, k] = np.angle(harmonic_vals)

        return harmonics, amps, phases , f0
 
    @classmethod
    def polar(cls, img, center=None, ell=[1, 1],angle = None, output_shape = None, show=False):
        """
        Convert image to elliptical-polar coordinates.
        Uses the *new* center after rotating with resize=True.
    
        Parameters
        ----------
        img : array
            Input grayscale image (H, W).
        center : tuple (x, y) or None
            Ellipse center in pixel coords. If None, estimated via fitEllipse.
        ell : [a, b]
            Scale factors for the x and y semi-axes.
        show : bool
            If True, show diagnostic plots.
        """
        import cv2
        
        if angle is None:
            try:
                mask = cls.mask(img)  
                contour = cls._set_contour(mask) 
                contour = np.array(contour, dtype=np.float32)
                contour_cv = contour[:, ::-1]  # (y,x) -> (x,y) 
                
                if center is None:
                    center_xy, _, angle = cv2.fitEllipse(contour_cv)
                else:
                    _, _, angle = cv2.fitEllipse(contour_cv)
                    center_xy = tuple(center)
            
                cx, cy = float(center_xy[0]), float(center_xy[1])
                
            except IndexError:
                print("Wrong mask sended. Please check Documentation, or write to @JBGiordano")
                sys.exit()
                
        else:
            angle = angle   # :)
            if center is None:
                center_xy = tuple(center)
                cx, cy = float(center_xy[0]), float(center_xy[1])
            else:
                raise ValueError("Angle and not center is not a compatible option")
            
        img_r = rotate(img,
                       angle=angle,
                       center=(cx, cy),     
                       resize=True,
                       order=1)

        if isinstance(ell, list) and len(ell) == 2:
            a, b = ell
        else:
            raise TypeError("ell must be a list as [y/y_max, x/x_max]")
        
        mask2 = cls.mask(img_r)       # TODO: must be a wiser way
        center2 = cls.center(mask2)
        
        ep_img = cls.warp_polar2(img_r,
                                    center=[center2[1], center2[0]],
                                    ell = ell, 
                                    output_shape = output_shape)
        
        if show:
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(2, 2)
            
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(img, cmap="gray")
            ax0.axis("off")
            ax0.scatter(cx, cy, s=30, c="r")
            ax0.set_title("Original")
            
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(img_r, cmap="gray")
            ax1.axis("off")
            ax1.scatter(center2[1], center2[0], s=30, c="r")
            ax1.set_title("Rotated (with new center)")
            
            ax2 = fig.add_subplot(gs[1, :])
            ax2.imshow(ep_img, cmap="gray", extent = (0,ep_img.shape[1], 0,360))
            ax2.set_title("Rotated → Elliptical coordinates")
            ax2.set_ylabel(r"$\theta$ (°)")
            ax2.set_xlabel("r (u.a.)")
            
            plt.tight_layout()
    
        return ep_img
    
    @classmethod
    def warp_polar2(cls,img, center,radius = None , ell = [1,1] , output_shape = None, **kwargs):
        """
        Elliptic-polar warp using skimage.transform.warp.
        Ellipses (x/a)^2 + (y/b)^2 = const become horizontal lines.
        
        Parameters
        ----------
        img : ndarray
            Input image.
        center : (cy, cx)
            Center of concentric ellipses (in pixels).
        radius : int
            Maximum radius for interpolate, according to the image
        ell: [y/y_max, x/x_max]
            Semi-axes scaling factors for y and x.
        output_shape : (H, W)
            Shape of the warped image (rows=r, cols=theta).
            Default is arround [360, 1093]
        """
        
        cy, cx = center
        
        if radius is None:
            radius = np.max(
                [np.hypot(cx,cy), 
                np.hypot(cx-  img.shape[0]-1, cy), 
                np.hypot(cx, cy - img.shape[1]-1), 
                np.hypot(cx - img.shape[0]-1, cy - img.shape[1]-1)
                ])
            
        if output_shape is None:
            height = 360
            width = int(np.ceil(radius))
            output_shape = (height, width)  
        else:
            height = output_shape[0]
            width = output_shape[1]
        
        k_angle = height /(2*np.pi)         # TODO: how many output rows per radian
        k_radius = width / radius           # TODO: how many output columns per unit radius
        
        warp_args = {"k_angle": k_angle, "k_radius": k_radius, "center": center, "ell": ell}
        map_func = cls._linear_polar_mapping2
        
        warped = warp(
                img, map_func, map_args=warp_args, output_shape=output_shape, **kwargs
                )
    
        return warped
    
    @classmethod
    def _linear_polar_mapping2(cls,output_coords, k_angle, k_radius, center, ell):
        
        """Inverse mapping for elliptical polar transform.
    
        Parameters
        ----------
        output_coords : (M, 2) ndarray
            Array of (row, col) coordinates in the output image.
            row → angle, col → radius.
        k_angle : float
            Scaling from output rows → angle.
        k_radius : float
            Scaling from output cols → radius.
        center : (cy, cx)
            Center of ellipses.
        ell : list
            Semi-axes scaling factors for x and y.
        """
        a, b = ell
        cy, cx = center
    
        radius = output_coords[:, 0] / k_radius   # cols → r
        angle  = output_coords[:, 1] / k_angle    # rows → θ
    
        rr = cy + b * radius * np.sin(angle)  # y (row)
        cc = cx + a * radius * np.cos(angle)  # x (col)
        return np.column_stack((rr, cc))
    
    @classmethod
    def _set_contour(cls, mask, show_cont = False):
        '''
        description
        '''
        inv_mask = np.logical_not(mask)
        label_img = label(inv_mask)
        props = regionprops(label_img)
        n_rows, n_cols = mask.shape
        hole_regions = []
        for region in props:
            min_row, min_col, max_row, max_col = region.bbox
            if min_row > 0 and min_col > 0 and max_row < n_rows and max_col < n_cols:
                hole_regions.append(region)
        if not hole_regions:
            return None
        largest_hole = max(hole_regions, key=lambda r: r.area)
        hole_mask = (label_img == largest_hole.label)
        contours = find_contours(hole_mask, level=0.5)
        if contours and show_cont:
            plt.imshow(mask, cmap='gray')
            plt.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2, color='red')
            plt.axis('off')
            plt.show()
        return contours[0] if contours else None

