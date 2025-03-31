import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
base_dir = os.path.dirname(__file__)
from scipy.interpolate import RegularGridInterpolator
from pyfcd.fcd_copy import compute_height_map

N = 500  
Lx = Ly = 1*np.pi
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

#h = 200*np.exp(-(X-5)**2) * np.exp(-(Y-5)**2)
#h = 100*np.sin(X*2)+100*np.cos(Y*2)
h = 0.1*np.sin(3*X)

h_hat = np.fft.fft2(h)

kx = (2 * np.pi / Lx) * np.fft.fftfreq(N, d=Lx/N)
ky = (2 * np.pi / Ly) * np.fft.fftfreq(N, d=Ly/N)
KX, KY = np.meshgrid(kx, ky, indexing='ij')

h_x_hat = 1j * KX * h_hat  
h_y_hat = 1j * KY * h_hat  

h_x = np.fft.ifft2(h_x_hat).real
h_y = np.fft.ifft2(h_y_hat).real

kx, ky = 10*2, 10*2 
I0 = np.sin(kx * X) * np.sin(ky * Y)

r = np.stack((X, Y), axis=-1)
H = 1       #altura del agua
u = -1*np.stack((h_x, h_y), axis=-1)
#u = -H*(1-(1.0002926 / 	1.3330))*np.stack((h_x, h_y), axis=-1) 
r_prim = (r - u)  

interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N) 

Ii = I.T

Ii = np.flip(Ii, axis=(0, 1))  # Invierte en ambas direcciones
#square_size=3.18-2.82
square_size = 0.63- 0.47
values = compute_height_map(I0, Ii, square_size=3.18-2.82, height=1)     #tuple = (height_map, phases, calibration_factor)
# #%%
# plt.figure()
# plt.imshow(I0, cmap="gray", origin="lower", extent=[0, Lx, 0, Ly])
# plt.title(r"Patrón Sinusoidal $I_0$")
# plt.show()

#%%
alturas = np.linspace(0.4, 1, 6)  # 6 valores entre 0 y 1
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 filas × 3 columnas

fig.tight_layout(pad=4.0)
plt.subplots_adjust(bottom=0.2)  # Espacio para el colorbar
vmin, vmax = np.inf, -np.inf
all_data = []


for idx, altura in enumerate(alturas):
    row, col = idx // 3, idx % 3 

    h = altura * np.sin(3 * X) 
    h_hat = np.fft.fft2(h)
    
    kx = (2 * np.pi / Lx) * np.fft.fftfreq(N, d=Lx/N)
    ky = (2 * np.pi / Ly) * np.fft.fftfreq(N, d=Ly/N)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    h_x_hat = 1j * KX * h_hat  
    h_y_hat = 1j * KY * h_hat  
    h_x = np.fft.ifft2(h_x_hat).real
    h_y = np.fft.ifft2(h_y_hat).real

    r = np.stack((X, Y), axis=-1)
    u = -1 * np.stack((h_x, h_y), axis=-1)
    r_prim = (r - u)  
    
    interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
    I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N) 
    Ii = np.flip(I.T, axis=(0, 1))
    
    square_size = 0.63 - 0.47
    values = compute_height_map(I0, Ii, square_size=square_size, height=1)
    data = values[0]/np.max(values[0]) - h.T/np.max(h.T)
    all_data.append(data)
    vmin = min(vmin, np.min(data))
    vmax = max(vmax, np.max(data))

plt.show()

# Plotear todos los subplots con la misma escala
for idx, (altura, data) in enumerate(zip(alturas, all_data)):
    row, col = idx // 3, idx % 3
    im = axs[row, col].imshow(
        data,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,  # Misma escala para todos
        origin='lower',
        extent=[0, Lx, 0, Ly]
    )
    axs[row, col].set_title(f"Altura = {altura:.2f}", fontsize=12)

# Colorbar común abajo
cax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]
fig.colorbar(im, cax=cax, orientation='horizontal', label='Intensidad')

plt.show()


#%%%

fig, axs = plt.subplots(2,2, figsize = (10,10))
axs[0,0].set_title('1*sin(3x), integrado', fontsize = 10)
axs[0,0].imshow(h.T, cmap = 'magma', origin="lower", extent=[0, Lx, 0, Ly])
im = axs[0, 0].imshow(h.T, cmap = 'magma', origin="lower", extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[0, 0])

axs[0,1].imshow(Ii, cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
axs[0,1].set_title(r"$I_0 (r - u(r))$ integrado" + f' ; {square_size:.2f}', fontsize = 10)
im = axs[0,1].imshow(Ii, cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[0, 1])

axs[1,0].set_title(f'Mapa devuelto por la fcd ; {square_size:.2f}', fontsize = 10)
axs[1,0].imshow(values[0], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
im = axs[1,0].imshow(values[0], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[1, 0])

axs[1,1].set_title(f'Resta funciones ; {square_size:.2f}', fontsize = 10)
axs[1,1].imshow(values[0]/np.max(values[0])- h.T/np.max(h.T),extent=[0, Lx, 0, Ly])
im = axs[1,1].imshow(values[0]/np.max(values[0])- h.T/np.max(h.T),cmap = 'magma', origin = 'lower', extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[1, 1])

#%%



# #%%
# plt.figure()
# plt.imshow(u[..., 0], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
# plt.colorbar(label=r'$u_x$')
# plt.title(r'Componente $u_x$ del desplazamiento')

# plt.figure()
# plt.imshow(u[..., 1], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
# plt.colorbar(label=r'$u_y$')
# plt.title(r'Componente $u_y$ del desplazamiento')

# plt.show()
