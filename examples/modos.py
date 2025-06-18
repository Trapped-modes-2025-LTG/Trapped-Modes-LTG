import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import numpy as np
import matplotlib.pyplot as plt

#%% Cargar carpeta de mediciones ancladas

base_dir = os.path.dirname(__file__)

map_folder = os.path.join(base_dir, "Pictures", "29_05", "maps")

calibration_files = [f for f in os.listdir(map_folder) if 'calibration_factor' in f and f.endswith('.npy')]

calibration_path = os.path.join(map_folder, calibration_files[0])
calibration_factor = np.load(calibration_path)

file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])
#%%

num_blocks = 64
blocks_per_row = int(np.sqrt(num_blocks))
img_size = 1024
block_size = img_size // blocks_per_row
n_modes = 4

#%% Cargar modo 0 para obtener CERO
amp0_list = []
phase0_list = []

for idx in range(num_blocks):
    _, a, p = analyze.block_amplitude(map_folder, block_index=idx, mode=1)
    amp0_list.append(a[:, :, 0])
    phase0_list.append(p[:, :, 0])
    print(f"Modo 0 - bloque {idx+1}/{num_blocks}")

# Convertir a arrays completos
amp0_imgs = np.zeros((img_size, img_size))
phase0_imgs = np.zeros((img_size, img_size))

for idx in range(num_blocks):
    i = idx // blocks_per_row
    j = idx % blocks_per_row
    amp0_imgs[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = amp0_list[idx]
    phase0_imgs[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = phase0_list[idx]
# Construir matriz CERO (modo 0 espacial completo)
CERO = amp0_imgs * np.cos(phase0_imgs)

#%% Calcular modos 1 al n_modes con resta del modo 0
amp = []
phase = []

for idx in range(num_blocks):
    i = idx // blocks_per_row
    j = idx % blocks_per_row

    # Extraer bloque correspondiente de CERO
    #block_cero = CERO[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

    _, a, p = analyze.block_amplitude(
        map_folder,
        block_index=idx,
        mode=n_modes,
        zero=CERO
    )
    amp.append(a[:, :])
    phase.append(p[:, :])
    print(f"Modo {n_modes} - bloque {idx+1}/{num_blocks}")

#%% Reconstruir imágenes completas por modo
amp_imgs = [np.zeros((img_size, img_size)) for _ in range(n_modes)]
phase_imgs = [np.zeros((img_size, img_size)) for _ in range(n_modes)]

for idx in range(num_blocks):
    i = idx // blocks_per_row
    j = idx % blocks_per_row
    for mode in range(n_modes):
        amp_imgs[mode][i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = amp[idx][:,:,mode]
        phase_imgs[mode][i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = phase[idx][:,:,mode]

#%% Ploteo
fig, axes = plt.subplots(nrows=n_modes, ncols=2, figsize=(12, 4 * n_modes))

for mode in range(n_modes):
    ax_amp = axes[mode, 0]
    im_amp = ax_amp.imshow(amp_imgs[mode], cmap='inferno')
    ax_amp.set_title(f'Amplitud - Modo {mode}', fontsize=10)
    cbar = fig.colorbar(im_amp, ax=ax_amp)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.offsetText.set_fontsize(8)

    ax_phase = axes[mode, 1]
    im_phase = ax_phase.imshow(phase_imgs[mode], cmap='twilight')
    ax_phase.set_title(f'Fase - Modo {mode}', fontsize=10)
    cbar = fig.colorbar(im_phase, ax=ax_phase)
    cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
