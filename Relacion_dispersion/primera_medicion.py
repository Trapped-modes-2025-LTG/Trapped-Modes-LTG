#2.43cm
#1cm
import os
import sys
import numpy as np
from skimage import io
from pyfcd.fcd import compute_height_map

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

def load_image(path):
    return io.imread(path, as_gray=True).astype(np.float32)

reference_path = os.path.join(base_dir, '16_04', 'referencia.tif')
reference_image = load_image(reference_path)

displaced_dir = os.path.join(base_dir, '16_04', 'medicion5_pelotita_2', 'gri_20250416_171242_C1S0001')
layers = [[5.7e-2, 1.0003], [1.2e-2, 1.48899],[1.0e-2, 1.48899], [2.4e-2, 1.34], [80e-2, 1.0003]]
square_size = 0.0022

output_dir = os.path.join(displaced_dir, 'maps')
os.makedirs(output_dir, exist_ok=True)

calibration_saved = False  # flag para guardar solo una vez

for i, fname in enumerate(sorted(os.listdir(displaced_dir)), start=1):
    if fname.endswith('.tif') and 'referencia' not in fname:
        displaced_path = os.path.join(displaced_dir, fname)
        displaced_image = load_image(displaced_path)

        height_map, _, calibration_factor = compute_height_map(
            reference_image, displaced_image, square_size, layers
        )

        output_path = os.path.join(output_dir, fname.replace('.tif', '_map.npy'))
        np.save(output_path, height_map)

        # Guardar el calibration_factor una sola vez
        if not calibration_saved:
            calibration_path = os.path.join(output_dir, 'calibration_factor.npy')
            np.save(calibration_path, np.array([calibration_factor]))
            calibration_saved = True