import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar

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
    # i = idx // blocks_per_row
    # j = idx % blocks_per_row

    # Extraer bloque correspondiente de CERO
    #block_cero = CERO[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

    _, a, p = analyze.block_amplitude(
        map_folder,
        block_index=idx,
        mode=n_modes,
        zero=0)#CERO)
    
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

#%%

UNO = amp_imgs[1]
DOS = amp_imgs[2]
TRES = amp_imgs[3]


#%%


contours_file = sorted([f for f in os.listdir(map_folder) if f.endswith('_contours.npy') and 'calibration_factor' not in f])

cont = contours_file[0]
#%%
foto = os.path.join(base_dir, "Pictures", "29_05","0_5mm_circular_100ms_20250529_141304_C1S0001000380.tif")
cy, cx = analyze.confined_peaks(analyze.load_image(foto), smoothed = 15, percentage= 90)

plt.scatter(cy, cx, s=20, color='red')
#%%
centro1 = tuple(map(int, (cx, cy)))

# Aplicar transformaciones polares (ya que el centro es el mismo)
polar1 = warp_polar(UNO, center=centro1, scaling='linear')
polar2 = warp_polar(DOS, center=centro1, scaling='linear')
polar3 = warp_polar(TRES, center=centro1, scaling='linear')

pnan1 = np.where(polar1 != 0, polar1, np.nan)
pnan2 = np.where(polar2 != 0, polar2, np.nan)
pnan3 = np.where(polar3 != 0, polar3, np.nan)

# Crear figura con 3 filas y 2 columnas
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Lista de imágenes y sus transformadas
originales = [UNO, DOS, TRES]
polares = [pnan1, pnan2, pnan3]
titulos = ["UNO", "DOS", "TRES"]

for i in range(3):
    # Imagen original con centro
    axes[i, 0].imshow(originales[i], cmap='inferno')
    axes[i, 0].scatter(*centro1[::-1], c='r', s=10)  # invertir para (y, x)
    axes[i, 0].set_title(f"Imagen original - {titulos[i]}")
    axes[i, 0].axis('off')

    # Imagen polar
    axes[i, 1].imshow(polares[i], cmap='inferno', aspect='auto')
    axes[i, 1].set_title(f"Coordenadas polares - {titulos[i]}")
    axes[i, 1].set_xlabel("θ")
    axes[i, 1].set_ylabel("r")

plt.tight_layout()
plt.show()

#%%

for i in range(360):
    val = np.nanmean(DOS[i, :])



#%%

# perfiles en tita = 150
plt.figure(figsize=(10, 4))

theta = np.arange(pnan1.shape[1])  # eje horizontal: columnas

#plt.plot(theta, pnan1[150, :], label='UNO', color='tomato')
plt.plot(theta, pnan2[150, :], label='DOS', color='gold')
plt.plot(theta, pnan3[150, :], label='TRES', color='limegreen')

plt.title("Corte radial en tita = 150 px")
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
exp2 = pnan2[150, 400:]
exp3 = pnan3[150, 400:]

from scipy.signal import find_peaks

peaks2, _ = find_peaks(exp2)  # podés ajustar height
peaks3, _ = find_peaks(exp3)


#%%
R = np.linspace(0, len(exp2), len(exp2))
plt.plot(R, exp2)
plt.title("exp2 en r = 150 y θ > 400")
plt.xlabel("θ")
plt.ylabel("Intensidad")
plt.grid(True)
plt.show()




























#%%
from scipy.signal import hilbert, find_peaks
import matplotlib.pyplot as plt

# Envolvente de Hilbert
env = np.abs(hilbert(exp2))

# Encontrar picos en la envolvente
peaks2h, _ = find_peaks(env)  # podés ajustar height


#%%"
# Graficar
plt.figure(figsize=(10, 4))
plt.plot(theta_valid, exp2_valid, label='Señal original')
plt.plot(theta_valid, env, label='Envolvente de Hilbert', linestyle='--')
plt.scatter(theta_valid[peaks2], env[peaks2], color='red', label='Picos')
plt.legend()
plt.xlabel("θ")
plt.ylabel("Intensidad")
plt.title("Picos en la envolvente")
plt.grid(True)
plt.tight_layout()
plt.show()




#%%
from scipy.optimize import curve_fit





# Modelo exponencial
def expon(x, A, k, C):
    return A * np.exp(-k * x) + C

# θ y valores en los picos
theta_peaks = theta_valid[peaks2]
env_peaks = env[peaks2]

# Ajuste
popt, pcov = curve_fit(expon, , env_peaks, p0=[1, 0.01, 0])

# Evaluar el fit
fit_vals = modelo_exp(theta_peaks, *popt)








# Graficar el ajuste
plt.figure(figsize=(6, 4))
plt.scatter(theta_peaks, env_peaks, label='Picos (envolvente)', color='red')
plt.plot(theta_peaks, fit_vals, label=f'Fit: A·exp(-kθ)+C\nA={popt[0]:.2f}, k={popt[1]:.4f}, C={popt[2]:.2f}', color='blue')
plt.xlabel("θ")
plt.ylabel("Intensidad de envolvente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title("Ajuste exponencial a los picos")
plt.show()



















