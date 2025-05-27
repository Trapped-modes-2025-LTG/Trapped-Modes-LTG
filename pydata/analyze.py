'''

'''

import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skimage import io
from pyfcd.fcd import fcd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.filters import gaussian#, sobel
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter

class analyze:
    @classmethod
    def load_image(cls,path):
        return io.imread(path, as_gray=True).astype(np.float32)

    @classmethod
    def mask(cls,image,smoothed, percentage,  sigma_background=100, alpha=0, show = False):

        def _subtract_background():
            background = gaussian(image.astype(np.float32), sigma=sigma_background, preserve_range=True)
            corrected = image.astype(np.float32) - alpha * background
            return corrected

        def _find_large_contours( binary):
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
        
        
        def _mostrar_resultados(binary, contornos, threshold, imagen_contorno):
            plt.figure()

            plt.subplot(1,3, 1)
            plt.imshow(image, cmap='gray')
            for c in contornos:
                plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
            plt.title("Original + contornos")
            plt.axis('off')


            plt.subplot(1,3, 2)
            plt.imshow(smooth, cmap='gray')
            plt.title("Suavizado")
            plt.axis('off')

            plt.subplot(1,3, 3)
            plt.imshow(imagen_contorno, cmap='gray')
            for c in contornos:
                plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
            plt.title("Binarizado")
            plt.axis('off')

            print(f"Cantidad de contornos detectados: {len(contornos)}")

            plt.tight_layout()
            plt.show()
        
        corrected = _subtract_background()
        smooth = uniform_filter(corrected, size=smoothed)
            
        threshold = np.mean(smooth)
        binary = (smooth > threshold).astype(np.uint8) * 255
        contornos = _find_large_contours(binary)
        imagen_contorno = binary
            
        if show:
            _mostrar_resultados( smooth, contornos, 0, imagen_contorno)
            
        return imagen_contorno, contornos
        
        mask = binary.astype(bool)

        return mask  

    @classmethod
    def folder(cls, reference_path, displaced_dir, layers, square_size, mask=False):
        reference_image = cls.load_image(reference_path)

        output_dir = os.path.join(displaced_dir, 'maps')
        os.makedirs(output_dir, exist_ok=True)

        calibration_saved = False

        for fname in sorted(os.listdir(displaced_dir)):
            if fname.endswith('.tif') and 'reference' not in fname:
                displaced_path = os.path.join(displaced_dir, fname)
                displaced_image = cls.load_image(displaced_path)

                if mask:        # TODO: unavailable yet, wrong call
                    displaced_image = displaced_image*cls.mask(displaced_image)

                height_map, _, calibration_factor = fcd.compute_height_map(
                    reference_image, displaced_image, square_size, layers
                )

                output_path = os.path.join(output_dir, fname.replace('.tif', '_map.npy'))
                np.save(output_path, height_map)

                if not calibration_saved:
                    calibration_path = os.path.join(output_dir, 'calibration_factor.npy')
                    np.save(calibration_path, np.array([calibration_factor]))
                    calibration_saved = True
                    
    @staticmethod
    def video(maps_dir, calibration_factor = None):
        
        if calibration_factor is None:
            calibration_path = os.path.join(maps_dir, 'calibration_factor.npy')
            if not os.path.isfile(calibration_path):
                raise FileNotFoundError(f"calibration_factor.npy was not found in {maps_dir}")
            calibration_factor = np.load(calibration_path).item()
        elif not isinstance(calibration_factor, (float, int)):
            raise TypeError("calibration_factor must be an integer or float")
            
        file_list = sorted([f for f in os.listdir(maps_dir) if f.endswith('.npy') 
                            and 'calibration_factor' not in f])
        df = pd.DataFrame({'filename': file_list})

        df['full_path'] = df['filename'].apply(lambda x: os.path.join(maps_dir, x))
        df['data'] = df['full_path'].apply(np.load)

        min_val = min([np.min(data) for data in df['data']])
        max_val = max([np.max(data) for data in df['data']])

        height_map = df['data'].iloc[0]
        n_rows, n_cols = height_map.shape
        x = np.linspace(0, n_cols, n_cols) * calibration_factor
        y = np.linspace(0, n_rows, n_rows) * calibration_factor
        x_mesh, y_mesh = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        im = ax.contourf(x_mesh * 1e3, y_mesh * 1e3, height_map * 1e3, 100,cmap='magma', vmin=min_val * 1e3, vmax=max_val * 1e3)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Height [mm]', rotation=270, labelpad=15)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_aspect("equal")

        def update(i):
            ax.clear()
            cont = ax.contourf(x_mesh * 1e3, y_mesh * 1e3, df['data'].iloc[i] * 1e3, 100,cmap='magma', vmin=min_val * 1e3, vmax=max_val * 1e3)
            ax.set_title(f'Frame {i}')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_aspect("equal")
            return cont.collections

        ani = animation.FuncAnimation(fig, update, frames=len(df), blit=False, interval=100)
        base_dir = os.path.dirname(__file__)
        output_path = os.path.join(base_dir, 'video.mp4')
        ani.save(output_path, writer='ffmpeg', fps=30)

        print(f"Saved in: {output_path}")



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
