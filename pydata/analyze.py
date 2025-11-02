'''
This script contains several functions to analyze measured data.
All of them are included in a single class called `analyze` for organizational purposes.
Each function is explained when it is called.
'''
import re
from collections import defaultdict, OrderedDict
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
import cv2
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
        
        calibration_saved = False
        file_list = sorted(os.listdir(displaced_dir))
        tif_list = [f for f in file_list if f.endswith('.tif') and 'reference' not in f]
        
        if polar:
            output_dir = os.path.join(displaced_dir, 'maps_polar')
        else:
            output_dir = os.path.join(displaced_dir, 'maps')
        
        os.makedirs(output_dir, exist_ok=True)
    
        existing_maps = sorted(f for f in os.listdir(output_dir) if f.endswith('_map.npy'))
        start_index = len(existing_maps)
    
        centers_path = os.path.join(output_dir, 'centers.txt')
        factor_path = os.path.join(output_dir, 'factor.txt')
            
        if smoothed:
            if not os.path.exists(centers_path):
                open(centers_path, "w").close()

            with open(centers_path, "r") as f:
                lines = f.readlines()
            start_index = max(start_index, len(lines))
    
            if show_mask:
                displaced_path = os.path.join(displaced_dir, tif_list[min(10, len(tif_list)-1)])
                displaced_image = cls.load_image(displaced_path)
    
                while True:
                    mask = cls.mask(displaced_image, smoothed=smoothed, show_mask=True)
                    plt.pause(8)
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
    

        for i, fname in tqdm(
            enumerate(tif_list[start_index:], start=start_index),
            total=len(tif_list) - start_index):
            
            displaced_path = os.path.join(displaced_dir, fname)
            displaced_image = cls.load_image(displaced_path)
    
            mask_applied = False
            
            if smoothed:
                mask = cls.mask(displaced_image, smoothed=smoothed)
                image_to_use = np.where(mask == 1, reference, displaced_image)
    
                center = cls.center(mask)    
                mask_applied = True
                
                if polar:
                    import cv2
                    contour = cls._set_contour(mask) 
                    contour = np.array(contour, dtype=np.float32)
                    contour_cv = contour[:, ::-1]  # (y,x) -> (x,y) 
                    _, _, angle = cv2.fitEllipse(contour_cv)
                    
                        
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
                height_map = height_map.astype(np.float32)
                np.save(output_path, height_map)
            else:
                if not os.path.exists(factor_path):
                    open(factor_path, "w").close()

                with open(factor_path, "r") as f:
                    lines = f.readlines()
                start_index = max(start_index, len(lines))
                output_path = os.path.join(output_dir, f"{base_name}_map_polar.npy")

                height_map_polar, factor = cls.polar(img = height_map, center = [center[1], center[0]],angle = angle,**kwargs)
                height_map_polar = height_map_polar.astype(np.float32)

                np.save(output_path, height_map_polar)
                with open(factor_path, "a") as f:  # append
                    f.write(f"{i}\t{factor}\n")
                    
            if smoothed:         
                with open(centers_path, "a") as f:  # append
                    f.write(f"{i}\t{center}\n")
            
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
        
        if isinstance(ell, list) and len(ell) == 2:
            a, b = ell
        else:
            raise TypeError("ell must be a list as [y/y_max, x/x_max]")
        
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
            if center is not None:
                center_xy = tuple(center)
                cx, cy = float(center_xy[0]), float(center_xy[1])
            else:
                raise ValueError("Angle and not center is not a compatible option")
        
        if angle > 180:
            angle = 180-angle
        
        img_r = rotate(img,
                       angle=angle,    
                       resize=True,
                       order=1)
        
        shape_img = img.shape        # assuming square
        shape_img_r = img_r.shape
        
        center2 = cls._rotate_center(y = cy, 
                                     x = cx, 
                                     angle_deg = angle, 
                                     shape = shape_img, 
                                     shape_rot = shape_img_r
                                     )
        coords = cls._bordes(img_r)
        ys, xs = zip(*coords)                 
        ys = np.array(ys, dtype=float)
        xs = np.array(xs, dtype=float)
        cy2, cx2 = center2 
        dists = np.hypot(xs - cx2, ys - cy2)
        imax  = int(np.argmax(dists))
        #y_far, x_far = ys[imax], xs[imax]
        dmax = float(dists[imax])
        
        ep_img = cls.warp_polar2(img_r,
                                 center=[center2[1], center2[0]],
                                 radius= dmax,
                                 ell = ell, 
                                 output_shape = output_shape
                                 )
        
        dist = cls._first_non_zero(ep_img)
        
        factor = dist / dmax
        
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
    
        return ep_img, factor
    
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
    
    @staticmethod
    def _rotate_center(y, x, angle_deg, shape, shape_rot):
        """
        Rotate a point (y, x) by `angle_deg` around the image center.
        Used after resize = True in polar()
        
        Parameters
        ----------
        y : float
            Center's y value.
        x : float
            Center's x value.
        angle_deg : float
            Angle used for rotation
        shape: (H,W)
            shape of the original image.
        shape_rot : (H, W)
            shape of the rotated image (after resize = True)
        """
        
        angle_rad = np.deg2rad(angle_deg)

        cy, cx = np.array(shape) / 2

        y0, x0 = y - cy, x - cx

        y1 =  np.cos(angle_rad) * y0 - np.sin(angle_rad) * x0
        x1 =  np.sin(angle_rad) * y0 + np.cos(angle_rad) * x0

        cy_r, cx_r = np.array(shape_rot) / 2
        
        return y1 + cy_r, x1 + cx_r 
    
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
    
    @classmethod
    def _bordes(cls, mapa):
        '''
        description
        '''
        H, W = mapa.shape
        sides = {
            "L": (slice(None), 0),      # x = 0
            "B": (H-1, slice(None)),    # y = H-1
            "R": (slice(None), W-1),    # x = W-1
            "T": (0, slice(None)),      # y = 0
        }
        out = []
        for name in ["L", "B", "R", "T"]:
            yy, xx = sides[name]
            edge = mapa[yy, xx]          
            idx = np.flatnonzero(edge)
            
            if idx.size == 0:
                out.append(None)
                continue

            k = int(round(idx.mean()))
            if name in ("L", "R"):
                y = k
                x = 0 if name == "L" else W-1
            else:
                y = H-1 if name == "B" else 0
                x = k
            out.append((y, x))

        return out
    
    @classmethod
    def _first_non_zero(cls, img):

        M = np.abs(img)           
        col_any = M.any(axis=0)        
        nz_cols = np.flatnonzero(col_any)
        if nz_cols.size == 0:
            return None                       
        x = int(nz_cols[-1])                 
        #ys = np.flatnonzero(M[:, x])           
        #y = int(ys)
        dist = x
        return dist


class ffts:
    
    import re

    '''

    =============================
    Requirement:  radios_npys must be a folder of .npy files, named like:
        
        r_8mm_a1679_t1s_20_h67_C1S0003_cal0.00022558593749999996_f768.2496931285473.npy

    =============================
    Imports: 

        import matplotlib
        matplotlib.use("TkAgg")          # set GUI backend (Big plots in terminal)
        import matplotlib.pyplot as plt
        import os
        import sys
        import numpy as np
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))
            # set your path to "analyze.py"
        from pydata.analyze import ffts

    =============================
    Examples to use ffts:
        1a) Create "fft_results_averaged" folder, with .npy files as

         {
          'r_0mm_a1679_t1s_0_h47.npy':   (freqs0, mean0, ses0),
          'r_2mm_a1679_t1s_0_h47.npy':   (freqs2, mean2, ses2),
          'r_4mm_a1679_t1s_0_h47.npy':   (freqs4, mean4, ses4),
          ...
         }

        1b) Way to load files 

        2) Fit curves, using saved "fft_results_averaged" folder

    ============(1a)============
       
    colormap = "magma"
    heights = ["_h47_", "_h54_", "_h67_"]
    floaters = ["_0_", "_10_", "_15_", "_20_"]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_subfolder = "radios_npys"
    search_path = os.path.join(base_dir, search_subfolder)
    
    save_dir = os.path.join(base_dir, "fft_results_averaged")
    os.makedirs(save_dir, exist_ok=True)
    
    for htag in heights:
        for etag in floaters:
            _ , averaged = ffts.process(htag, etag, search_path)
    
            # all_averaged[(etag, htag)] = averaged
    
            plt.figure(figsize=(6,4))
            colors = plt.cm.get_cmap(colormap, len(averaged))
    
            for i, (base, (freqs, mean,ses, useful)) in enumerate(
                sorted(
                    (kv for kv in averaged.items()),
                    # if any(s in kv[0] for s in ("_10mm_", "_12mm_", "_14mm_"))),
                    key=lambda kv: ffts.extract_radii(kv[0])
                )
            ):
    
    	    # TODO: Replace errors if its neccesary 
                label = f"r={ffts.extract_radii(base):.0f} mm"
                plt.errorbar(freqs, 
    			mean, 
    			yerr=ses, 
    			fmt='.-', 
    			capsize=3, 
    			alpha=0.85, 
    			color=colors(i), 
    			label=label)
    
            plt.xlim(0, 12) 
            plt.xlabel("Frecuencia (Hz)")
    
            plt.title(f"Floater {etag} - Height {htag}")
            plt.legend(fontsize=8, ncol=2)
            plt.grid( linestyle='--', alpha=0.5)
            plt.tight_layout()	
            safe_tag = f"{etag.strip('_')}_{htag.strip('_')}"
            plt.savefig(f"{safe_tag}.pdf")
            plt.show()
        
            flat = {}
            for base, (freqs, mean, ses, useful) in averaged.items():
                flat[f"{base}_freqs"]  = freqs
                flat[f"{base}_mean"]   = mean
                flat[f"{base}_ses"]    = ses
            
            np.savez(os.path.join(save_dir, f"fft_results_averaged_{safe_tag}.npz"), **flat)
    
    ============(1b)============

    # Change both .npy file name (1st one) and npy key (2nd one)
    # if it's needed

    data = load_and_plot("fft_results_averaged_0_h47.npz")
    freqs, mean, ses = data['r_12mm_a1679_t1s_0_h47.npy']

    ============(2)=============

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))
    from pydata.analyze import ffts
    
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    
    save_dir   = os.path.join(base_dir, "fft_results_averaged")
    results_dir = os.path.join(base_dir, "ajustes_resultados")
    
    archivos_ts = [0, 10, 15, 20]
    
    file_paths = [os.path.join(save_dir, f"fft_results_averaged_{t}_h47.npz") for t in archivos_ts]
    
    radios = ["2", "4", "6"]
    
    fit = ffts.fit_me(radios, archivos_ts, file_paths, results_dir)


    '''

    dt = 1 / 125

    @classmethod    
    def extract_cal(cls, name: str) -> float:
        
        cal_pat = re.compile(r"_cal(\d+\.\d+)_")          # captures "0.000767"
        m = cal_pat.search(name)
        if not m:
            raise ValueError(f"Could not find cal in: {name}")
        return float(m.group(1))
    
    @classmethod    
    def extract_fac(cls,name: str) -> float:

        fac_pat = re.compile(r"_f(\d+(?:\.\d+)?)")       # captures "756.1381"
        m = fac_pat.search(name)
        if not m:
            raise ValueError(f"Could not find factor in: {name}")
        return float(m.group(1))
    
    @classmethod    
    def extract_radii(cls,key: str) -> float:

        rad_pat = re.compile(r"_(\d+(?:\.\d+)?)mm_")
        m = rad_pat.search(key)
        if m:
            return float(m.group(1))
        raise ValueError(f"Cannot parse radius from filename: {key}")
    
    @classmethod    
    def strip_group_tokens(cls,name: str) -> str:
        stem, ext = os.path.splitext(name)
    
        # 1) remove replica token: _C1S0001_ (and if it appears right before the extension)
        stem = re.sub(r"_C1S\d{4}_", "_", stem)
        stem = re.sub(r"_C1S\d{4}(?=$)", "", stem)
    
        # 2) remove calibration block: _cal<decimal>_
        #    (your files look like _cal0.00022558593749999996_)
        stem = re.sub(r"_cal\d+(?:\.\d+)?_", "_", stem)
    
        # 3) remove factor block: _f<number>_  OR _f<number> just before the extension
        #    (your example ends with ..._f768.2433822186998.npy)
        stem = re.sub(r"_f\d+(?:\.\d+)?_", "_", stem)           # with trailing underscore
        stem = re.sub(r"_f\d+(?:\.\d+)?(?=$)", "", stem)        # right before .npy
    
        return stem + ext


    @classmethod    
    def fft_radii(cls,arr, path):
        # arr: (n_times, n_angles) = (t, ang)
    
        R = cls.extract_radii(path)
        cal = cls.extract_cal(path)
        fac = cls.extract_fac(path)
    
        n_times, n_angles = arr.shape
        N = n_times
    
        fft_vals = np.fft.fft(arr, axis=0) / N
        fft_vals[0, :] = 0
    
        freqs = np.fft.fftfreq(N, d=cls.dt)
        pos_mask = freqs >= 0
    
        fft_abs_pos = np.abs(fft_vals)[pos_mask, :]  # (N_pos, n_angles)
    
        for i in range(n_angles):
            fft_angle = fft_abs_pos[:, i]
            fft_angle_max = np.max(fft_angle)
            if fft_angle_max > 0:                     # <-- minimal guard
                fft_angle = fft_angle / fft_angle_max
            else:
                fft_angle = 0.0 * fft_angle           # no signal on this angle
            fft_abs_pos[:, i] = fft_angle
    
        mean_over_angles = np.mean(fft_abs_pos, axis=1)
    
        n_non_nans = np.count_nonzero(fft_abs_pos[3, :] != 0)
        r_px = (R / (cal * fac))
        n_radii = int(2 * np.pi * r_px)
    
        if n_radii == 0:
            useful = False
            denom = 1
        elif 0 < n_radii < 360:
            useful = False
            denom = np.sqrt(n_radii)
        else:
            useful = True
            # <-- minimal guard to avoid denom == 0 if no angles are active
            denom = np.sqrt(n_radii * (n_non_nans / 360)) if n_non_nans > 0 else 1
    
        # TODO primera std sobre las medias de los ángulos
        std_over_angles = np.std(fft_abs_pos, axis=1) / denom
    
        return freqs[pos_mask], mean_over_angles, std_over_angles, useful   
    
    @classmethod    
    def process(cls, height_tag, floater_tag, search_path):  
        """
        Defined for each water height and floater 
        """
    
        npy_files = [
            f for f in os.listdir(search_path)
            if f.endswith(".npy") and floater_tag in f and height_tag in f
        ]
        
        npy_files.sort(key=cls.extract_radii)  # sort by radius
        
        results = OrderedDict()  # keep insertion order = radius order
        
        for f in npy_files:
            arr = np.load(os.path.join(search_path, f))
            freqs, mean, std, useful = cls.fft_radii(arr, f)
            results[f] = (freqs, mean, std, useful)
    
        # group replicates by  S####.npy
        grouped = defaultdict(list)
        for fname, (freqs, mean, ses, useful) in results.items():
            base = cls.strip_group_tokens(fname)
            grouped[base].append((freqs, mean, ses, useful))
    
        # average across replicates
        averaged = OrderedDict()
        EPS = 1e-8  # <-- piso para evitar SE=0 -> pesos inf
    
        for base in sorted(grouped.keys(), key=cls.extract_radii):
            triplets = grouped[base]
            freqs0 = triplets[0][0]                      # (F,)
            means  = np.stack([t[1] for t in triplets])  # (R,F)
            ses    = np.stack([t[2] for t in triplets])  # (R,F)
    
            # --- parche mínimo: evitar 0/inf/NaN en SE antes de los pesos
            ses = np.where(~np.isfinite(ses), np.nan, ses)
            ses = np.where(ses <= 0, EPS, ses)
    
            with np.errstate(divide='ignore', invalid='ignore'):
                # pesos: w_i = 1 / SE_i^2
                w = 1.0 / np.square(ses)                 # (R,F)
                w_sum = np.nansum(w, axis=0)             # (F,)
                num   = np.nansum(w * means, axis=0)     # (F,)
    
                # si w_sum==0 en algún bin, usar fallback sin pesos para NO dejar NaNs
                fallback_mean = np.nanmean(means, axis=0)
                fallback_se   = np.nanstd(means, axis=0) / np.sqrt(np.sum(np.isfinite(means), axis=0).clip(min=1))
    
                mean_w = np.divide(
                    num, w_sum,
                    out=fallback_mean,                   # <-- fallback
                    where=w_sum > 0
                )
    
                se_w = np.divide(
                    1.0, np.sqrt(w_sum),
                    out=fallback_se,                     # <-- fallback
                    where=w_sum > 0
                )
    
            useful_vals = [t[3] for t in triplets]       # booleans
            useful_red = all(useful_vals)                
    
            averaged[base] = (freqs0, mean_w, se_w, useful_red)
    
        return results, averaged

    # 2 

    @classmethod
    def fit_me(cls, radios, archivos_ts, file_paths, results_dir, polgrad = 1, f_min = 4.25, f_max = 5.75):

        from scipy.optimize import curve_fit
    
        def lorentz(f, A1, A2, gamma1, gamma2, f1, f2, B):
            return A1 * (0.5 * gamma1)**2 / ((f - f1)**2 + (0.5 * gamma1)**2) \
                 + A2 * (0.5 * gamma2)**2 / ((f - f2)**2 + (0.5 * gamma2)**2) + B
    
        os.makedirs(results_dir, exist_ok=True)
    
        for i, path in enumerate(file_paths):
            data = cls._load_and_plot(path)  # { base: (freqs, mean, ses) }
            file_name = os.path.splitext(os.path.basename(path))[0]
            path_dir = os.path.join(results_dir, file_name)
            os.makedirs(path_dir, exist_ok=True)
    
            t1s_val = archivos_ts[i]
    
            for r in radios:
                key = f"r_{r}mm_a1679_t1s_{t1s_val}_h47.npy"  # <-- FIXED f-string
    
                if key not in data:
                    print(f"[!] Key {key} didn't find in {file_name}")
                    continue
    
                freqs, mean, ses = data[key]

                # guards
                if freqs is None or freqs.size == 0 or mean is None or mean.size == 0:
                    print(f"[!] Empty data for {key} in {file_name}")
                    continue
    
                # fit window
                mask_fit   = (freqs >= f_min) & (freqs <= f_max)
                f_fit      = freqs[mask_fit]
                amps_fit   = mean[mask_fit]
                sigmas_fit = ses[mask_fit] if ses is not None else np.ones_like(amps_fit)
    
                if f_fit.size < 5:
                    print(f"[!] Too few points in window for {key} in {file_name}")
                    continue
    
                # detrend
                coef = np.polyfit(f_fit, amps_fit, polgrad)
                poly_trend = np.polyval(coef, f_fit)
                amps_fit_detrended = amps_fit - poly_trend
    
                # peaks
                picos, _ = find_peaks(amps_fit_detrended, threshold=0.0005)
                if picos.size < 2:
                    print(f"[!] Not enough peaks for {key} in {file_name}")
                    continue
    
                A1_0 = amps_fit_detrended[picos[0]]
                A2_0 = amps_fit_detrended[picos[1]]
                f1_0 = f_fit[picos[0]]
                f2_0 = f_fit[picos[1]]
                gamma1_0 = (f_fit[picos[1]] - f_fit[picos[0]])/1.7
                gamma2_0 = gamma1_0
    
                B_0 = 0.0
                # keep your original p0 order if you prefer (works with your lorentz def)
                p0 = [A1_0, A2_0, f1_0, f2_0, gamma1_0, gamma2_0, B_0]
    
                try:
                    popt, pcov = curve_fit(
                        lorentz, f_fit, amps_fit_detrended,
                        p0=p0, sigma=sigmas_fit,
                        absolute_sigma=True, maxfev=20000
                    )
                except Exception as e:
                    print(f"[x] Falló el ajuste para {key} en {file_name}: {e}")
                    continue
    
                # plot + save
                f_plot = np.linspace(f_fit.min(), f_fit.max(), 500)
                fit_curve = lorentz(f_plot, *popt)
    
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                               gridspec_kw={'height_ratios': [3, 1]})
                ax1.errorbar(f_fit, amps_fit_detrended, yerr=sigmas_fit,
                             fmt='.', color='gray', alpha=0.7, label='Datos')
                ax1.plot(f_plot, fit_curve, 'r-', lw=2, label='Ajuste Lorentziano')
                ax1.set_ylabel('Amplitud')
                ax1.set_title(f"{file_name} – {key}")
                ax1.legend(loc='best', fontsize=8, frameon=True)
                ax1.grid(True, ls='--', alpha=0.5)
    
                residuals = amps_fit_detrended - np.interp(f_fit, f_plot, fit_curve)
                ax2.axhline(0, color='black', lw=1)
                ax2.plot(f_fit, residuals, 'o', color='blue', markersize=3)
                ax2.errorbar(f_fit, residuals, yerr=sigmas_fit, alpha=0.7)
                ax2.set_xlabel('Frecuencia [Hz]')
                ax2.set_ylabel('Residuo')
                ax2.grid(True, ls='--', alpha=0.5)
    
                plt.tight_layout()
                fig_path = os.path.join(path_dir, f"{key}.png")
                plt.savefig(fig_path, dpi=200)
                plt.close(fig)
    
                param_names = ['A1', 'A2', 'f1', 'f2', 'γ1', 'γ2', 'B']
                txt_path = os.path.join(path_dir, f"{key}_params.txt")
                with open(txt_path, "w") as f:
                    for name, val in zip(param_names, popt):
                        f.write(f"{name} = {val:.6g}\n")
   
    @classmethod
    def _load_and_plot(cls, npz_path):
    
        npz = np.load(npz_path)
        bases = [k[:-6] for k in npz.files if k.endswith("_freqs")]
        if not bases:
            print(f"[!] No series found in {os.path.basename(npz_path)}")
            return {}
    
        def sort_key(base):
            try:
                return ffts.extract_radii(base)
            except Exception:
                return float("+inf")
    
        bases = sorted(bases, key=sort_key)
    
        out = {}
        for base in bases:
            freqs = npz[base + "_freqs"]
            mean  = npz[base + "_mean"]
            ses   = npz[base + "_ses"]  # present in your saved schema
            out[base] = (freqs, mean, ses)
        return out

