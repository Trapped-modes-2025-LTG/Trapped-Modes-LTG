import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze

base_dir = os.path.dirname(__file__)

tif_folder = os.path.join(base_dir, "Pictures", "mask")

reference_path = os.path.join(base_dir, "Pictures", "reference_df.tif")

layers = [[3.2e-2,1.0003], [ 1.2e-2,1.48899], [3.4e-2,1.34], [ 80e-2 ,1.0003]]

analyze.folder(reference_path, tif_folder, layers, 0.002, smoothed=15, percentage =95 , timer= False)





base_dir = os.path.dirname(os.path.dirname(__file__))
maps_dir = os.path.join(base_dir, 'Pictures', 'mask', 'maps')

calibration_files = [f for f in os.listdir(maps_dir) if 'calibration_factor' in f and f.endswith('.npy')]
if not calibration_files:
    raise FileNotFoundError("No se encontró el archivo de calibration factor en la carpeta.")

calibration_path = os.path.join(maps_dir, calibration_files[0])
calibration_factor = np.load(calibration_path)
print("Calibration factor encontrado:", calibration_factor)


file_list = sorted([f for f in os.listdir(maps_dir) if f.endswith('.npy') and 'calibration_factor' not in f])


tiempos = [i * (1 / 500) for i in range(len(file_list))]

data = [
    calibration_factor * np.load(os.path.join(maps_dir, f))
    for f in file_list
]


df = pd.DataFrame({'tiempo': tiempos, 'data': data})


#%%
matriz = df.loc[0, 'data'] 

plt.imshow(matriz, cmap='viridis') 
plt.colorbar(label='Altura (calibrada)')
plt.title(f"Matriz en t = {df.loc[0, 'tiempo']:.3f} s")
plt.tight_layout()
plt.show()


# coordenadas del punto a seguir
i, j = 411,633

evolucion_punto = np.array([frame[i, j] for frame in df['data']])


plt.plot(df['tiempo'], evolucion_punto)
plt.xlabel("Tiempo [s]")
plt.ylabel(f"Valor en ({i}, {j})")
plt.title(f"Evolución temporal del punto ({i}, {j})")
plt.grid(True)
plt.tight_layout()
plt.show()

N = len(evolucion_punto)
dt = df['tiempo'][1] - df['tiempo'][0]


fft_vals = fft(evolucion_punto)
frecuencias = fftfreq(N, dt)

frecuencias_pos = frecuencias[:N//2]
amplitudes_pos = np.abs(fft_vals)[:N//2]


plt.figure(figsize=(8, 4))
plt.plot(frecuencias_pos, amplitudes_pos)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.title(f"Espectro temporal en el punto ({i}, {j})")
plt.grid(True)
plt.tight_layout()
plt.show()


peaks, _ = find_peaks(amplitudes_pos)


pico_principal_idx = peaks[np.argmax(amplitudes_pos[peaks])]
frecuencia_dominante = frecuencias_pos[pico_principal_idx]
amplitud_dominante = amplitudes_pos[pico_principal_idx]


plt.figure(figsize=(8, 4))
plt.plot(frecuencias_pos, amplitudes_pos, label='Espectro')
plt.plot(frecuencia_dominante, amplitud_dominante, 'ro', label=f'Pico principal: {frecuencia_dominante:.2f} Hz')

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.title(f"Espectro en el punto ({i}, {j})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# n = 3  # numero de modos a guardar
# T = len(df)
# Nx, Ny = df['data'][0].shape
# print(f"Datos cargados desde df: T={T}, Nx={Nx}, Ny={Ny}")

# amplitudes_n = np.zeros((T, n))
# fases_n = np.zeros((T, n))
# modos_indices = None

# for t in range(T):
#     matriz = df['data'][t]

#     # FFT2 centrada
#     fft2 = np.fft.fft2(matriz)
#     fft2_shifted = np.fft.fftshift(fft2)

#     # Aplanar
#     amplitudes_flat = np.abs(fft2_shifted).flatten()
#     fases_flat = np.angle(fft2_shifted).flatten()

#     if t == 0:
#         # Elegimos los n modos con mayor amplitud en t=0
#         indices_ordenados = np.argsort(amplitudes_flat)[::-1]
#         modos_indices = indices_ordenados[:n]

#     # Guardamos amplitudes y fases de los mismos n modos
#     amplitudes_n[t, :] = amplitudes_flat[modos_indices]
#     fases_n[t, :] = fases_flat[modos_indices]

# plt.figure(figsize=(10, 5))
# for i in range(n):
#     plt.plot(df['tiempo'], amplitudes_n[:, i], label=f"Modo {i}")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Amplitud")
# plt.title(f"Evolución de los {n} modos principales")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# for i in range(n-1):
#     plt.plot(df['tiempo'], fases_n[:, i], label=f"Modo {i}")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Fase [rad]")
# plt.title(f"Evolución de las fases de los {n} modos principales")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
