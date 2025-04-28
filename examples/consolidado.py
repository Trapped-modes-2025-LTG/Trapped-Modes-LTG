import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from skimage.io import imread
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from pydata.test_deteccion import deteccion
from pyfcd.fcd import compute_height_map

frames_dir = os.path.join(BASE_DIR,'') # complete with frames path
image_path2 = os.path.join(BASE_DIR,'')  # reference image
reference = imread(image_path2).astype('double')

# TODO: length and refraction index of every layer from bottom to top
#              'air'             'acrylic'       'water'            'air'
layers = [[5.7e-2, 1.0003], [1.2e-2, 1.48899], [4.3e-2, 1.34], [80e-2, 1.0003]]
square_size = 0.022

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.bmp', '.tif'))])
height_maps = []

for fname in frame_files:
    print(f"Procesando {fname}...")
    image_path = os.path.join(frames_dir, fname)
    image_raw = imread(image_path).astype('double')
    masked_image = deteccion(image_raw)[0]
    mask = deteccion(image_raw)[1]
    combinada = np.where(mask, masked_image, reference)
 #   out = compute_height_map(reference, masked_image, square_size, layers)
    out = compute_height_map(reference, combinada, square_size, layers)
    out_masked = np.zeros_like(reference)
    out_masked[mask] = out[0][mask]
    height_maps.append(out_masked)
