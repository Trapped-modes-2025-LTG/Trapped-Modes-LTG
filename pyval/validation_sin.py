import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
base_dir = os.path.dirname(__file__)
from scipy.interpolate import RegularGridInterpolator
from pyfcd.fcd import compute_height_map
import matplotlib.cm as cm
import matplotlib.colors as mcolors

N = 500  
Lx = Ly = 1*np.pi
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')
#%%
#h = 200*np.exp(-(X-5)**2) * np.exp(-(Y-5)**2)
#h = 100*np.sin(X*2)+100*np.cos(Y*2)
alt = 0.1
h = alt*np.sin(3*X)

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

#%%%
norm_hT = h.T/np.max(h.T)
fig, axs = plt.subplots(2,2, figsize = (10,10))
axs[0,0].set_title(f'{alt}*sin(3x), integrado', fontsize = 10)
axs[0,0].imshow(norm_hT, cmap = 'magma', origin="lower", extent=[0, Lx, 0, Ly])
im = axs[0, 0].imshow(norm_hT, cmap = 'magma', origin="lower", extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[0, 0])

axs[0,1].imshow(Ii, cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
axs[0,1].set_title(r"$I_0 (r - u(r))$ integrado" , fontsize = 10)
im = axs[0,1].imshow(Ii, cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[0, 1])

meanh = (np.max(values[0]) + np.min(values[0]))*0.5
difh = (np.max(values[0]) - np.min(values[0]))*0.5
values_media = (values[0]-meanh)/difh

axs[1,0].set_title('Mapa fcd: centrado y normalizado', fontsize = 10)
axs[1,0].imshow(values_media, cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
im = axs[1,0].imshow(values_media, cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[1, 0])

axs[1,1].set_title('Resta funciones ', fontsize = 10)
axs[1,1].imshow(values_media- norm_hT,extent=[0, Lx, 0, Ly])
im = axs[1,1].imshow(values_media- norm_hT,cmap = 'magma', origin = 'lower', extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=axs[1, 1])

#%%   Generación de los mapas

alturas = np.linspace(0.01, 1, 20)  # 6 valores entre 0 y 1
vmin, vmax = np.inf, -np.inf
all_data = []

Hs = np.linspace(1,10,5)

for i in range(len(Hs)):
    pass

    for idx, altura in enumerate(alturas):
        row, col = divmod(idx, 3)
    
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
        u = -Hs[i] * np.stack((h_x, h_y), axis=-1)
        r_prim = (r - u)  
        
        interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
        I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N) 
        Ii = np.flip(I.T, axis=(0, 1))
        
        #square_size = 0.63 - 0.47
        square_size = Hs[i]*0.1
        values = compute_height_map(I0, Ii, square_size=square_size, height=Hs[i])
        
        meanh = (np.max(values[0]) + np.min(values[0]))*0.5
        difh = (np.max(values[0]) - np.min(values[0]))*0.5
        values_media = (values[0]-meanh)/difh
        
        data = values_media - h.T/np.max(h.T)
        all_data.append(data)
        vmin = min(vmin, np.min(data))
        vmax = max(vmax, np.max(data))
    
    fila_corte = N // 2  # Índice de la fila central
    
    cmap = cm.gnuplot  # Colormap 'magma'
    #norm = mcolors.Normalize(vmin=np.min(alturas), vmax=np.max(alturas))  # Normalización
    norm = mcolors.Normalize(vmin = 0, vmax = 1)  # Normalización: son las alturas
    
    fig, ax = plt.subplots(figsize=(8, 6))
    #rec = []
    for altura, data in zip(alturas, all_data):
        x_vals = np.linspace(0, Lx, N)  # Eje X
        y_vals = data[fila_corte, :]  # Corte en la fila central
        color = cmap(norm(altura))  # Color según altura
    
        ax.plot(x_vals, y_vals, color=color, label=f"H = {altura:.2f}")
    
    # Crear la barra de color
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Altura")
    
    ax.set_xlabel("x")
    ax.set_ylabel("Intensidad")
    ax.set_title("Cortes en la fila central para distintas alturas \n "+ f'Altura del agua {Hs[i]}')
    ax.grid('--', alpha = 0.5)
    ax.plot()
# plt.imshow(u[..., 1], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
# plt.colorbar(label=r'$u_y$')
# plt.title(r'Componente $u_y$ del desplazamiento')

# plt.show()
