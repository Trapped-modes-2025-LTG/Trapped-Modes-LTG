import sys
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq, ifft2, fft2

#%%
# Ruta base y carpeta de mapas
base_dir = os.path.dirname(os.path.dirname(__file__))
maps_dir = os.path.join(base_dir, '16_04', 'medicion5_pelotita_2', 'gri_20250416_171242_C1S0001', 'maps')

# Cargar calibration factor
calibration_path = os.path.join(maps_dir, 'calibration_factor.npy')
calibration_factor = np.load(calibration_path).item()  # .item() si está guardado como array con un solo valor

# Crear DataFrame con archivos .npy (ignorando calibration_factor)
file_list = sorted([f for f in os.listdir(maps_dir) if f.endswith('.npy') and 'calibration_factor' not in f])[250:400]

df = pd.DataFrame({'filename': file_list})

df['full_path'] = df['filename'].apply(lambda x: os.path.join(maps_dir, x))
df['data'] = df['full_path'].apply(np.load)

# min_val = min([np.min(data) for data in df['data']])
# max_val = max([np.max(data) for data in df['data']])

df['diagonal'] = df['data'].apply(np.diagonal)
#%%
N = 1024
x = np.linspace(0,N*1e3*calibration_factor*np.sqrt(2),N)
frame = 100
alturas = df['diagonal'].iloc[frame]

fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].plot(x,alturas*1e3, '.-')
ax[0].grid(linestyle = '--',alpha = 0.5)
ax[0].set_xlabel('Posición')
ax[0].set_ylabel('Altura')
ax[0].set_title('Altura a t fijo')

k1 = fftfreq(N, N*1e3*calibration_factor*np.sqrt(2)/N)

altura_fft = fft(alturas)
ax[1].plot(k1, np.abs(np.real(altura_fft)), '.-')
ax[1].set_title(f'Transformada a t = {frame/500}s')
ax[1].grid(linestyle = '--',alpha = 0.5)
ax[1].set_xlabel(r'$k$')
ax[1].set_xlim(-0.4, 0.4)

#plt.savefig('relacion_disp_t_fijo.png', bbox_inches = 'tight')


#%%
alturas_array = np.stack(df['diagonal'].values)  # shape: (T, N)
#%% Frecuencia temporal

altura_r = alturas_array[:,100]
t = np.linspace(0, alturas_array.shape[0]/500, alturas_array.shape[0])

w1 = fftfreq(alturas_array.shape[0], d = 1/500)

altura_r_fft = np.real(fft(altura_r))

fig, ax = plt.subplots(1,2, figsize = (10,5))

ax[0].plot(t,altura_r*1e3, '.-')
ax[0].grid(linestyle = '--',alpha = 0.5)
ax[0].set_xlabel('Tiempo')
ax[0].set_ylabel('Altura')
ax[0].set_title('Altura a r fijo')

k1 = fftfreq(N, N*1e3*calibration_factor*np.sqrt(2)/N)

altura_fft = np.real(fft(alturas))
ax[1].plot(w1, np.abs(altura_r_fft), '.-')
ax[1].set_title(f'Transformada a r = {100*calibration_factor*1e3*np.sqrt(2):.2f}mm')
ax[1].grid(linestyle = '--',alpha = 0.5)
ax[1].set_xlabel(r'$\omega$')

#plt.savefig('relacion_disp_r_fijo.png', bbox_inches = 'tight')
#%%
N = alturas_array.shape[1]  
T = alturas_array.shape[0]  

x = np.linspace(0, N, N)* calibration_factor * np.sqrt(2)  
t = np.linspace(0, (T)/500, T) 

X, T_mesh = np.meshgrid(x, t)
plt.figure()
plt.pcolormesh(X, T_mesh, alturas_array)  
plt.xlabel('Posición (m)')
plt.ylabel('Tiempo (s)')
plt.colorbar(label='Altura (m)')
plt.tight_layout()
#plt.savefig('relacion_dispersion_mapa_rt.png', bbox_inches = 'tight')
#%%
h_xt = alturas_array  # en metros

# FFT y amplitud
fft_result = np.fft.fft2(h_xt)
fft_shifted = np.fft.fftshift(fft_result)
amplitud = np.abs(np.real(fft_shifted))

# Ejes frecuenciales
nt, nx = h_xt.shape
dt = np.max(t) / len(t)
dx = N * calibration_factor * np.sqrt(2) / N  # mm

freqs_t = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))  # ciclos por segundo
freqs_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # ciclos por m

# Convertir a unidades físicas (rad/s y rad/mm)
omega = 2 * np.pi * freqs_t
k_vals = 2 * np.pi * freqs_x
K, W = np.meshgrid(k_vals, omega)

# Gráfico
plt.figure()
plt.pcolormesh(-K, W, amplitud, shading='auto')
# plt.colorbar()
plt.plot([0, 0], [-10000, 10000], 'k-', linewidth=0.7)
plt.plot([-1300, 1300], [0, 0], 'k-', linewidth=0.7)

plt.xlabel(r'$k$ [rad/m]', fontsize=16)
plt.ylabel(r'$\omega$ [rad/s]', fontsize=16)
plt.tick_params(axis='both', direction='in')

# Relación de dispersión (en rad/s y rad/mm)
g = 9.81              # m/s^2
sigma = 0.0728      # mN/m
rho = 1002              # kg/m^3
H = 0.024             # m

k = np.linspace(0, 1500, 200000)  # rad/mm
w = np.sqrt(k * (g + (sigma * k**2) / rho) * np.tanh(k * H))
plt.plot(k, w, 'r-', label='Relación \n gravito-capilar', linewidth = 2)
plt.legend(fontsize=12)
plt.ylim(-300,300)
plt.xlim(-1000,1000)
#plt.savefig('mapa_relacion_dispersion.png', bbox_inches='tight')