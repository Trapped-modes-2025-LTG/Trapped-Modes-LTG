'''

'''

import os
import numpy as np
from skimage import io
from pyfcd.fcd import compute_height_map
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class analyze:
    @classmethod
    def load_image(cls,path):
        return io.imread(path, as_gray=True).astype(np.float32)

    @classmethod
    def mask(cls,image):
        return image  

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

                if mask:
                    displaced_image = cls.mask(displaced_image)

                height_map, _, calibration_factor = compute_height_map(
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


