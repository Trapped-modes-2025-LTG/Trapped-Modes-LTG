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
import cv2
#from skimage.filters import sobel
from skimage import io
from skimage.draw import polygon
from skimage.measure import regionprops, label,find_contours
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

class analyze:
    @classmethod
    def load_image(cls,path):
        '''
        Loads a grayscale image and converts it to float32.
    
        Parameters
        ----------
        path: str - Path to the image file.
    
        Returns
        -------
        ndarray - 2D grayscale image as a float32 NumPy array.
        '''
        return io.imread(path, as_gray=True).astype(np.float32)

    @classmethod 
    def side(cls,x,y,binary):
        h, w = binary.shape
        x = int(x)
        y = int(y)
        if y == 0 or y == w-1: 
            if binary[y, x+1] == 0: 
                return "right"
            else:
                return 'left'
        elif x == 0 or x == h-1:
            if binary[y+1,x] == 0:
                return "up"
            else:
                return "down"
            
    @classmethod 
    def borders(cls, side, ps, maxim):
        x0, y0 = ps[0]
        x1, y1 = ps[1]
        x2, y2 = ps[2]
        x3, y3 = ps[3]
        
        if side == "right":
            xs_derecha = [
                p[1] for i, p in enumerate(ps)
                if i % 2 == 0 and p[0] == y0 and p[1] > x0
            ]
        
            if xs_derecha:
                x_stop_val = min(xs_derecha)

                for i, p in enumerate(ps):
                    if i % 2 == 0 and p[0] == y0 and p[1] == x_stop_val:
                        del ps[i:i+2]  # Eliminar el punto 
                        break
            
                x_stop = x_stop_val - 1
                camino_x0 = np.linspace(x0, x_stop, int(x_stop - x0) + 1)
                camino_y0 = np.full_like(camino_x0, y0)
            else:
                pass
            #     x_stop = maxim
            #     camino_x0 = np.linspace(x0, x_stop, int(x_stop - x0) + 1)
            #     camino_y0 = np.full_like(camino_x0, y0)
        
            #     found = False
            #     segmento_x = []
            #     segmento_y = []
        
            #     if y0 == maxim - 1:
            #         for dy in range(1, maxim):
            #             new_y = y0 - dy
            #             if new_y < 0:
            #                 break
            #             if binary[new_y, x_stop] != 0:
            #                 segmento_x = [x_stop] * (dy + 1)
            #                 segmento_y = list(range(y0, new_y - 1, -1))
            #                 found = True
            #                 break
            #     else:
            #         for dy in range(1, height):
            #             new_y = y0 + dy
            #             if new_y >= height:
            #                 break
            #             if binary[new_y, x_stop] != 0:
            #                 segmento_x = [x_stop] * (dy + 1)
            #                 segmento_y = list(range(y0, new_y + 1))
            #                 found = True
            #                 break
        
            #     if found:
            #         camino_x0 = np.concatenate([camino_x0, segmento_x])
            #         camino_y0 = np.concatenate([camino_y0, segmento_y])
        
            plt.plot(camino_x0, camino_y0, lw=2, c="lime", zorder=10)
            pass
        return
    
    @classmethod
    def correct_contours(cls, contornos, binary):
        contornos = sorted(contornos, key=lambda c: len(c), reverse=True)
        
        threshold = 300 # TODO: ver como calcular esto
        
        inner_contours = []
        out_contours = []
        
        for c in contornos[:5]:
            p0 = c[0]
            p1 = c[-1]
            distancia = np.linalg.norm(p0 - p1)
        
            if distancia <= 1:
                inner_contours.append(c)
            else:
                out_contours.append(c)
        
        if len(inner_contours) > 1:
            inner_contours = inner_contours[:2]  
            if len(inner_contours[1]) <= threshold:
                inner_contours = [inner_contours[0]]
            else: 
                return inner_contours
        
        out_contours = [c for c in out_contours if len(c) >= threshold]
                
        if len(out_contours) == 1:
            cnt1 = out_contours[0].astype(np.int32)
            
            if not cnt1[-1][0] == cnt1[0][0] or cnt1[-1][1] == cnt1[0][1]:
            
                p1 = cnt1[0]
                p2 = cnt1[-1]
                
                y_max = binary.shape[0] - 1 
                
                caminos_y = []
                caminos_x = []
                    
                for y, x in [p1, p2]:
                    y, x = int(y), int(x)
                    
                    if x == 0:
                        if y > y_max / 2:
                            camino_y = np.linspace(y, y_max, int(y_max - y))
                        else:
                            camino_y = np.linspace(0, y, int(y))
                        camino_x = np.zeros_like(camino_y)
                    
                    elif y == 0:
                        if x > y_max / 2:
                            camino_x = np.linspace(x, y_max, int(y_max - x))
                        else:
                            camino_x = np.linspace(0, x, int(x))
                        camino_y = np.zeros_like(camino_x)
                
                    elif x == y_max:
                        if y > y_max / 2:
                            camino_y = np.linspace(y, y_max, int(y_max - y))
                        else:
                            camino_y = np.linspace(0, y, int(y))
                        camino_x = np.full_like(camino_y, y_max)
                
                    elif y == y_max:
                        if x > y_max / 2:
                            camino_x = np.linspace(x, y_max, int(y_max - x))
                        else:
                            camino_x = np.linspace(0, x, int(x))
                        camino_y = np.full_like(camino_x, y_max)
                    
                    caminos_y.append(camino_y)
                    caminos_x.append(camino_x)
        
            cnt2 = inner_contours[0]
            
            camino1 = np.stack([caminos_y[0], caminos_x[0]], axis=1).astype(np.int32)
            camino2 = np.stack([caminos_y[1], caminos_x[1]], axis=1).astype(np.int32)
                
            cnt1 = np.concatenate([cnt1, camino1, camino2])
 
            return cnt1, cnt2
        
        elif len(out_contours) == 2:
            
            ps = []         # [  [yi0, xi0], [yf0, xf0]  ,  [yi1, xi1], [yf1, xf1]  ]
            # plt.figure()
            # plt.imshow(binary)

            for c in out_contours:
                cix = c[:, 1][0]
                ciy = c[:, 0][0]
                cfx = c[:, 1][-1]
                cfy = c[:, 0][-1]
                ps.append([int(ciy), int(cix)])
                ps.append([int(cfy), int(cfx)])

                plt.scatter(c[:, 1], c[:, 0], s = 1, c = "red")
                plt.scatter(c[:, 1][0], c[:, 0][0], s = 100, c = "cyan")
                plt.scatter(c[:, 1][-1], c[:, 0][-1], s = 100, c = "black")
                
            y0, x0 = ps[0]
            
            side_p0 = cls.side(x0, y0, binary)
            
            maxim= binary.shape[0]-1
            
            # ordenar segun el sentido de las agujas del reloj
            ps = np.array(ps) 
            p0 = ps[0]
            resto = ps[1:]
            angulos = np.arctan2(resto[:, 0] - p0[0], resto[:, 1] - p0[1])
            indices_orden = np.argsort(angulos)
            resto_ordenado = resto[indices_orden]
            ps_o = np.vstack([p0, resto_ordenado])
            
            if side_p0 == "right":
                pass
            if side_p0 == "left":
                pass
            if side_p0 == "down": 
                pass
            if side_p0 == "down":
                pass
            
            # if side_p0 == "right":
            #     xs_derecha = [
            #         p[1] for i, p in enumerate(ps)
            #         if i % 2 == 0 and p[0] == y0 and p[1] > x0
            #     ]
            
            #     if xs_derecha:
            #         x_stop_val = min(xs_derecha)
    
            #         for i, p in enumerate(ps):
            #             if i % 2 == 0 and p[0] == y0 and p[1] == x_stop_val:
            #                 del ps[i:i+2]  # Eliminar el punto 
            #                 break
                
            #         x_stop = x_stop_val - 1
            #         camino_x0 = np.linspace(x0, x_stop, int(x_stop - x0) + 1)
            #         camino_y0 = np.full_like(camino_x0, y0)
            #     else:
            #         x_stop = maxim
            #         camino_x0 = np.linspace(x0, x_stop, int(x_stop - x0) + 1)
            #         camino_y0 = np.full_like(camino_x0, y0)
            
            #         height = binary.shape[0]
            #         found = False
            #         segmento_x = []
            #         segmento_y = []
            
            #         if y0 == height - 1:
            #             for dy in range(1, height):
            #                 new_y = y0 - dy
            #                 if new_y < 0:
            #                     break
            #                 if binary[new_y, x_stop] != 0:
            #                     segmento_x = [x_stop] * (dy + 1)
            #                     segmento_y = list(range(y0, new_y - 1, -1))
            #                     found = True
            #                     break
            #         else:
            #             for dy in range(1, height):
            #                 new_y = y0 + dy
            #                 if new_y >= height:
            #                     break
            #                 if binary[new_y, x_stop] != 0:
            #                     segmento_x = [x_stop] * (dy + 1)
            #                     segmento_y = list(range(y0, new_y + 1))
            #                     found = True
            #                     break
            
            #         if found:
            #             camino_x0 = np.concatenate([camino_x0, segmento_x])
            #             camino_y0 = np.concatenate([camino_y0, segmento_y])
            
            #     plt.plot(camino_x0, camino_y0, lw=2, c="lime", zorder=10)
            
            cont_unido = None
            cnt2 = inner_contours[0]
            print()
            return cont_unido, cnt2
        
    @classmethod
    def mask(cls,image,smoothed, percentage, sigma_background=100, show_mask = False):
        '''
        lipsum.
    
        Parameters
        ----------
        image: ndarray - 2D grayscale image.
        smoothed: float - .
        percentage: float - .
        sigma_background: int - 
        alpha: int - .
        show_mask: False : Bool - Decide if the proccess used is ploted or not.
        
        Returns
        -------
        mask : .
        '''

        def _find_large_contours(binary):
            contours = find_contours(binary, level=0.5)
            if percentage > 0:
                def area_contorno(contour):
                    x = contour[:, 1]
                    y = contour[:, 0]
                    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                areas = np.array([area_contorno(c) for c in contours])
                umbral = np.percentile(areas, percentage)
                return [c for c, a in zip(contours, areas) if a >= umbral]
            return contours
        
        def _mostrar_resultados(binary, contornos, imagen_contorno):
            plt.figure()

            plt.subplot(1,3, 1)
            plt.imshow(image, cmap='gray')
            for c in contornos:
                plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
            plt.title("Original + edges")
            plt.axis('off')

            plt.subplot(1,3, 2)
            plt.imshow(smooth, cmap='gray')
            plt.title("Smoothed")
            plt.axis('off')

            plt.subplot(1,3, 3)
            plt.imshow(imagen_contorno, cmap='gray')
            for c in contornos:
                plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
            plt.title("Binarized")
            plt.axis('off')

            print(f"Cantidad de contornos detectados: {len(contornos)}")

            plt.tight_layout()
            plt.show()
        
        smooth = uniform_filter(image, size=smoothed)
            
        threshold = np.mean(smooth)
        binary = (smooth > threshold).astype(np.uint8) * 255
        contornos = _find_large_contours(binary)
        imagen_contorno = binary
        if show_mask:
            _mostrar_resultados(binary, contornos, imagen_contorno)
        cnts = cls.correct_contours(contornos, binary)
    
        mask_shape = image.shape[:2]  
        outer_mask = np.zeros(mask_shape, dtype=np.uint16)
        inner_mask = np.zeros(mask_shape, dtype=np.uint16)
    
        cnt_outer = np.asarray(cnts[0], dtype=np.int32)
        cnt_inner = np.asarray(cnts[1], dtype=np.int32)
        
        if cnt_outer.ndim == 2:
            cnt_outer = cnt_outer[:, np.newaxis, :]
        if cnt_inner.ndim == 2:
            cnt_inner = cnt_inner[:, np.newaxis, :]
        
        cv2.drawContours(outer_mask, [cnt_outer], -1, color=1, thickness=cv2.FILLED)
        cv2.drawContours(inner_mask, [cnt_inner], -1, color=1, thickness=cv2.FILLED)
    
        between_mask = outer_mask - inner_mask
        between_mask[between_mask < 0] = 0
        mask = (1-between_mask).astype(float)
        
        contornos = [cnts[0], cnts[1]]
        
        if show_mask:
            _mostrar_resultados(binary, contornos, imagen_contorno)
        
        return mask, contornos

    @classmethod
    def folder(cls, reference_path, displaced_dir, layers, square_size,
           smoothed=None, percentage=None, sigma_background=100,
           show_mask=False, timer=False, only="mask"):
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
        percentage : int, optional
            Percentage threshold for contour masking.
        sigma_background : int, default=100
            Sigma for the background subtraction filter.
        show_mask : bool, default=False
            If True, displays the generated mask and contours.
        timer : bool, default=False
            If True, prints progress during processing.
        only : str, default="mask"
            Determines what to save: "mask", "contours", or "both".
    
        Returns
        -------
        None
            Saves height maps and/or contours in a "maps" folder inside displaced_dir.
        '''
        
        reference = cls.load_image(reference_path)
        output_dir = os.path.join(displaced_dir, 'maps')
        os.makedirs(output_dir, exist_ok=True)
    
        calibration_saved = False
        file_list = sorted(os.listdir(displaced_dir))
    
        for n, fname in enumerate(file_list):
            if not (fname.endswith('.tif') and 'reference' not in fname):
                continue
    
            displaced_path = os.path.join(displaced_dir, fname)
            displaced_image = cls.load_image(displaced_path)
    
            mask_applied = False
            mask = None
            cnt = None
    
            if smoothed and percentage:
                mask, cnt = cls.mask(
                    displaced_image,
                    smoothed,
                    percentage,
                    sigma_background=sigma_background,
                    show_mask=show_mask
                )
                displaced = mask.T * displaced_image
                displaced_w = np.where(mask.T, displaced, reference)
                image_to_use = displaced_w
                mask_applied = True
            else:
                image_to_use = displaced_image
    
            height_map, _, calibration_factor = fcd.compute_height_map(
                reference,
                image_to_use,
                square_size,
                layers
            )
    
            if mask_applied:
                height_map *= mask.T
    
            base_name = fname.replace('.tif', '')
    
            if only in {"mask", "both"}:
                output_path = os.path.join(output_dir, f"{base_name}_map.npy")
                np.save(output_path, height_map)
    
            if only in {"contours", "both"} and cnt is not None:
                sorted_cnts = sorted(cnt, key=lambda c: len(c), reverse=True)
                cnts_top3 = sorted_cnts[:3]
                contour_path = os.path.join(output_dir, f"{base_name}_contours.npy")
                np.save(contour_path, np.array(cnts_top3, dtype=object))
    
            if not calibration_saved:
                calibration_path = os.path.join(output_dir, 'calibration_factor.npy')
                np.save(calibration_path, np.array([calibration_factor]))
                calibration_saved = True
    
            if timer:
                print(f"{n + 1}/{len(file_list)}")
                    
    @staticmethod
    def video(maps_dir, calibration_factor=None, frame_final=300, n_extra_frames=20):
    

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
    def confined_peaks(cls, image, cnt= None, smoothed = 0, percentage = 80):
        if cnt is None:
            _, cnts = cls.mask(image,
                                smoothed = smoothed, 
                                percentage = percentage,
                                mask_save= False,
                                show_mask = False
                                )
            plt.figure()
            plt.imshow(image)
            for c in cnts:
                plt.scatter(c[:, 1], c[:, 0],s = 1, c = "red")
            cnt = cnts[1]       # TODO: select the correct index
        
        height, width = image.shape[:2]
            
        cnt = cnt.reshape(-1, 2)
        r = cnt[:, 1]
        c = cnt[:, 0]
            
        mask = np.zeros((height, width), dtype=np.uint16)
        rr, cc = polygon(r, c, mask.shape)
        mask[rr, cc] = 1
            
        labeled = label(mask)
        props = regionprops(labeled)
            
        if props:
            cy, cx = props[0].centroid
            return cy, cx
        else: 
            pass
     
    @staticmethod
    def block_amplitude(map_folder, f0=None, tasa=500, mode=1, num_blocks=64, block_index=0, t_limit=None, neighbor = None):

        file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])
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

            block = m[i*block_size : (i+1)*block_size,j*block_size : (j+1)*block_size]
            mask_block = mask_validos[i*block_size : (i+1)*block_size,j*block_size : (j+1)*block_size]

            block_masked = np.where(mask_block, block, np.nan)
            maps.append(block_masked)

        maps = np.stack(maps, axis=0)  # (N, block_size, block_size)
        maps = np.transpose(maps, (1, 2, 0))  # (block_size, block_size, N)

        ny, nx, N = maps.shape
        dt = 1 / tasa

        # === FFT ===
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
                return np.zeros(mode), np.full((ny, nx, mode), None, dtype=object), np.full((ny, nx, mode), None, dtype=object)
            max_peak_index = peaks[np.argmax(mean_spectrum[peaks])]
            f0 = fft_freqs[max_peak_index]

        # === Índices de armónicos ===
        harmonics = [f0 * n for n in range(1, mode+1)]
        indices = [np.argmin(np.abs(fft_freqs - f)) for f in harmonics]

        # === Amplitud y fase para cada armónico ===
        amps = np.zeros((ny, nx, mode))
        phases = np.zeros((ny, nx, mode))

        for k, idx in enumerate(indices):
            harmonic_vals = fft_vals[:, :, idx]
            amps[:, :, k] = 2 * np.abs(harmonic_vals) / N  # amplitud real (factor 2 por simetría)
            phases[:, :, k] = np.angle(harmonic_vals)

        return harmonics, amps, phases
    
    
    # if not cnt1[-1][0] == cnt1[0][0] or cnt1[-1][1] == cnt1[0][1]:
        
        #     p1 = cnt1[0]
        #     p2 = cnt1[-1]
            
        #     y_max = image.shape[0] - 1 
            
        #     caminos_y = []
        #     caminos_x = []
                
        #     for y, x in [p1, p2]:
        #         y, x = int(y), int(x)
                
        #         if x == 0:
        #             if y > y_max / 2:
        #                 camino_y = np.linspace(y, y_max, int(y_max - y))
        #             else:
        #                 camino_y = np.linspace(0, y, int(y))
        #             camino_x = np.zeros_like(camino_y)
                
        #         elif y == 0:
        #             if x > y_max / 2:
        #                 camino_x = np.linspace(x, y_max, int(y_max - x))
        #             else:
        #                 camino_x = np.linspace(0, x, int(x))
        #             camino_y = np.zeros_like(camino_x)
            
        #         elif x == y_max:
        #             if y > y_max / 2:
        #                 camino_y = np.linspace(y, y_max, int(y_max - y))
        #             else:
        #                 camino_y = np.linspace(0, y, int(y))
        #             camino_x = np.full_like(camino_y, y_max)
            
        #         elif y == y_max:
        #             if x > y_max / 2:
        #                 camino_x = np.linspace(x, y_max, int(y_max - x))
        #             else:
        #                 camino_x = np.linspace(0, x, int(x))
        #             camino_y = np.full_like(camino_x, y_max)
                
        #         caminos_y.append(camino_y)
        #         caminos_x.append(camino_x)          
# class ImageEnhancer:
#     def __init__(self, imagen, sigma_background=100, alpha=0):
#         self.image = imagen
#         self.sigma_background = sigma_background
#         self.alpha = alpha

#     def _subtract_background(self):
#         background = gaussian(self.image.astype(np.float32), sigma=self.sigma_background, preserve_range=True)
#         corrected = self.image.astype(np.float32) - self.alpha * background
#         return corrected

#     def _find_large_contours(self, binary, percentil_contornos=0):
#         contours = find_contours(binary, level=0.5)
#         if percentil_contornos > 0:
#             def area_contorno(contour):
#                 x = contour[:, 1]
#                 y = contour[:, 0]
#                 return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
#             areas = np.array([area_contorno(c) for c in contours])
#             umbral = np.percentile(areas, percentil_contornos)
#             return [c for c, a in zip(contours, areas) if a >= umbral]
#         return contours

#     # def _find_contours_by_sobel(self, image, levels=[0.1], percentil_contornos=0):
#     #     edges = sobel(image.astype(float) / 255.0)
#     #     contornos = []
#     #     for nivel in levels:
#     #         c = find_contours(edges, level=nivel)
#     #         contornos.extend(c)
#     #     if percentil_contornos > 0 and contornos:
#     #         def area_contorno(contour):
#     #             x = contour[:, 1]
#     #             y = contour[:, 0]
#     #             return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
#     #         areas = np.array([area_contorno(c) for c in contornos])
#     #         umbral = np.percentile(areas, percentil_contornos)
#     #         contornos = [c for c, a in zip(contornos, areas) if a >= umbral]
#     #     return contornos

#     def procesar(self, suavizado=5, mostrar=True, percentil_contornos=0):
    
#         corrected = self._subtract_background()
#         smooth = uniform_filter(corrected, size=suavizado)

#         # if metodo_contorno == "sobel":
#         #     contornos = self._find_contours_by_sobel(smooth, levels=[0.16], percentil_contornos=percentil_contornos)
#         #     imagen_contorno = sobel(smooth.astype(float) / 255.0)
#         # elif metodo_contorno == "binarizacion":
#         #     threshold = np.mean(smooth)
#         #     binary = (smooth > threshold).astype(np.uint8) * 255
#         #     contornos = self._find_large_contours(binary, percentil_contornos=percentil_contornos)
#         #     imagen_contorno = binary
#         # else:
#         #     raise ValueError(f"Método de contorno no reconocido: {metodo_contorno}")
        
#         threshold = np.mean(smooth)
#         binary = (smooth > threshold).astype(np.uint8) * 255
#         contornos = self._find_large_contours(binary, percentil_contornos=percentil_contornos)
#         imagen_contorno = binary
        
#         if mostrar:
#             self._mostrar_resultados( smooth,  imagen_contorno, contornos, 0, imagen_contorno)
        
#         return imagen_contorno, contornos
        
        

#     def _mostrar_resultados(self, smooth, binary, contornos, threshold, imagen_contorno):
#         plt.figure()

#         plt.subplot(1,3, 1)
#         plt.imshow(self.image, cmap='gray')
#         for c in contornos:
#             plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
#         plt.title("Original + contornos")
#         plt.axis('off')


#         plt.subplot(1,3, 2)
#         plt.imshow(smooth, cmap='gray')
#         plt.title("Suavizado")
#         plt.axis('off')

#         plt.subplot(1,3, 3)
#         plt.imshow(imagen_contorno, cmap='gray')
#         for c in contornos:
#             plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
#         plt.title("Binarizado")
#         plt.axis('off')

#         print(f"Cantidad de contornos detectados: {len(contornos)}")

#         plt.tight_layout()
#         plt.show()

# #%%      
# base_dir = os.path.dirname(os.path.dirname(__file__))

# tif_folder = os.path.join(base_dir, "datos","toroide_deteccion")

# tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith('.tif')]


# df_tif = pd.DataFrame({
#     'nombre_archivo': tif_files,
#     'ruta_completa': [os.path.join(tif_folder, f) for f in tif_files]
# })

# path = df_tif["ruta_completa"].iloc[5]
# imagen = analyze.load_image(path)
# #imagen = imread("C:/Users/Tomas/Desktop/FACULTAD/LABO 6/Resta-P8139-150Oe-50ms-1000.tif")[400:700, 475:825]
# enhancer = ImageEnhancer(imagen=imagen)

# enhancer = ImageEnhancer(imagen=imagen)
# binary, contornos = enhancer.procesar(
#     suavizado=20,
#     percentil_contornos=30
# )        
