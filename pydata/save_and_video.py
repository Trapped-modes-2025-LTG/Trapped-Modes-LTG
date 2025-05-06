'''
This script shows how to save the analyzed photos as ".npy" files in a dedicated folder, and how to create a video
from them. The two parts are separated into cells and are just an example of what we used.

For the first part, it is necessary to have a folder containing a reference image and several displaced images
of the pattern. The reference image should show the pattern with the free surface at rest. All images must be in
".tif" format. A folder named "maps" is created inside the original folder to store the analyzed arrays.

A "for" loop is used to analyze each image (previously binarized with "load_image(path)") using the main function
from this repository: "compute_height_map" from "fcd.py". This function takes the reference and displaced images,
the square size of the pattern, and the list of layers for the effective height as inputs. The last two are physical
parameters of the setup. "square_size" corresponds to half the wavelength (λ/2), and "layers" describes the sequence
of materials, as format "[[reflection_index1, height1], [...],...]" from the pattern to the camera.

The resulting height maps are saved as ".npy" files in the "maps" folder, along with a file named
"calibration_factor.npy", which contains the pixel-to-meter conversion factor.

Better methods to replace the "for" loop were not explored. Optimizing this could help reduce computation time.

In the second part of the script, the easiest way to create a video from the previous "maps" folder was developed.
The ".npy" files 

---

In the second part of the script, a simple way to create a video from the "maps" folder is shown. The ".npy" files
are loaded, and a "matplotlib" animation is generated using contour plots to visualize the height evolution.

---
Usage:
1. Place a reference image (at rest) and all displaced ".tif" images in a folder.
2. Update the variables "reference_path", "displaced_dir", "layers", and "square_size".
3. Run the first section to compute and save height maps in "maps/".
4. Run the second section to create a video from the maps.
'''


import os
import sys
import numpy as np
from skimage import io
from pyfcd.fcd import compute_height_map

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

def load_image(path):
    return io.imread(path, as_gray=True).astype(np.float32)

reference_path = os.path.join(base_dir, '.tif')              #Complete with your reference path
reference_image = load_image(reference_path)

displaced_dir = os.path.join(base_dir, '/')                  #Complete with your dispaced dir

layers = []                                                  #Complete with your layers
square_size = None                                           #Complete with your λ/2 

output_dir = os.path.join(displaced_dir, 'maps')
os.makedirs(output_dir, exist_ok=True)

calibration_saved = False

for i, fname in enumerate(sorted(os.listdir(displaced_dir)), start=1):
    if fname.endswith('.tif') and 'reference' not in fname:
        displaced_path = os.path.join(displaced_dir, fname)
        displaced_image = load_image(displaced_path)

        height_map, _, calibration_factor = compute_height_map(reference_image, displaced_image, square_size, layers)

        output_path = os.path.join(output_dir, fname.replace('.tif', '_map.npy'))
        np.save(output_path, height_map)

        if not calibration_saved:
            calibration_path = os.path.join(output_dir, 'calibration_factor.npy')
            np.save(calibration_path, np.array([calibration_factor]))
            calibration_saved = True
            
#%%

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

maps_dir = os.path.join(displaced_dir, 'maps')          

# Load calibration_factor
calibration_path = os.path.join(maps_dir, 'calibration_factor.npy')
calibration_factor = np.load(calibration_path).item()

# Create DataFrame with files '.npy'
file_list = sorted([f for f in os.listdir(maps_dir) if f.endswith('.npy') and 'calibration_factor' not in f])[250:1000]

df = pd.DataFrame({'filename': file_list})

# Read data
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
cbar.set_label('Altura [mm]', rotation=270, labelpad=15)
ax.set_xlabel('Posición x [mm]')
ax.set_ylabel('Posición y [mm]')
ax.set_aspect("equal")

def update(i):
    ax.clear()
    cont = ax.contourf(x_mesh * 1e3, y_mesh * 1e3, df['data'].iloc[i] * 1e3, 100,cmap='magma', vmin=min_val * 1e3, vmax=max_val * 1e3)
    ax.set_title(f'Frame {i}')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_aspect("equal")
    print(f'{i}/{len(df)}')
    return cont.collections

ani = animation.FuncAnimation(fig, update, frames=len(df), blit=False, interval=100)

output_path = os.path.join(base_dir, 'video2.mp4')
ani.save(output_path, writer='ffmpeg', fps=30)

print(f"Saved in: {output_path}")
