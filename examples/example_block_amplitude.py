import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

#%% Cargar carpeta de mediciones ancladas

base_dir = os.path.dirname(__file__)

map_folder = os.path.join(base_dir, "29_5", "med_S0001", "maps")

calibration_files = [f for f in os.listdir(map_folder) if 'calibration_factor' in f and f.endswith('.npy')]

calibration_path = os.path.join(map_folder, calibration_files[0])
calibration_factor = np.load(calibration_path)

file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])

#%% Generar las listitas
num_blocks = 64
blocks_per_row = int(np.sqrt(num_blocks))
img_size = 1024
block_size = img_size // blocks_per_row

amp = []
phase = []

for i in range(num_blocks):
    _, a, p = analyze.block_amplitude(map_folder, block_index=i, mode=3)
    amp.append(a[:, :])    
    phase.append(p[:, :])  
    
    print(f"{i+1}/{num_blocks}")
#%% plotear
amp_imgs = [np.zeros((img_size, img_size)) for _ in range(3)]
phase_imgs = [np.zeros((img_size, img_size)) for _ in range(3)]

for idx in range(num_blocks):
    i = idx // blocks_per_row
    j = idx % blocks_per_row

    for mode in range(3):
        amp_imgs[mode][
            i*block_size:(i+1)*block_size,
            j*block_size:(j+1)*block_size
        ] = amp[idx][:,:,mode]

        phase_imgs[mode][
            i*block_size:(i+1)*block_size,
            j*block_size:(j+1)*block_size
        ] = phase[idx][:,:,mode]

# Graficar cada modo
for mode in range(3):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(f'Amplitud - Modo {mode+1}')
    plt.imshow(amp_imgs[mode], cmap='inferno')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f'Fase - Modo {mode+1}')
    plt.imshow(phase_imgs[mode], cmap='twilight')
    plt.colorbar()

    plt.tight_layout()
    #plt.savefig(f"modo_{mode+1}_armonico_circular.pdf",bbox_inches = "tight")