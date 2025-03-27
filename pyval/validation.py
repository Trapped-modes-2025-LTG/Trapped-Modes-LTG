import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyfcd.fcd import compute_height_map
import matplotlib.pyplot as plt
base_dir = os.path.dirname(__file__)
from scipy.interpolate import RegularGridInterpolator

N = 500  
Lx = Ly = 10  
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

#h = 200*np.exp(-(X-5)**2) * np.exp(-(Y-5)**2)
#h = 100*np.sin(X*2)+100*np.cos(Y*2)
h = 10*np.sin(X)

h_hat = np.fft.fft2(h)

kx = (2 * np.pi / Lx) * np.fft.fftfreq(N, d=Lx/N)
ky = (2 * np.pi / Ly) * np.fft.fftfreq(N, d=Ly/N)
KX, KY = np.meshgrid(kx, ky, indexing='ij')

h_x_hat = 1j * KX * h_hat  
h_y_hat = 1j * KY * h_hat  

h_x = np.fft.ifft2(h_x_hat).real
h_y = np.fft.ifft2(h_y_hat).real

kx, ky = 10, 10  
I0 = np.sin(kx * X) * np.sin(ky * Y)

r = np.stack((X, Y), axis=-1)
H = 1       #altura del agua
u = -H*(1-(1.0002926 / 	1.3330))*np.stack((h_x, h_y), axis=-1) 
r_prim = (r - u)  

interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N) 


Ii = I.T

Ii = np.flip(Ii, axis=(0, 1))  # Invierte en ambas direcciones
values = compute_height_map(I0, Ii, square_size=1, height=1)     #tuple = (height_map, phases, calibration_factor)
#%%
plt.figure()
plt.imshow(I0, cmap="gray", origin="lower", extent=[0, Lx, 0, Ly])
plt.title(r"Patrón Sinusoidal $I_0$")
plt.show()

plt.figure()
plt.title(r'$h = 100 (sin(2x)+ cos(2y))$')
contour_h = plt.contourf(X, Y, h, 100, cmap="magma")

plt.figure()
plt.imshow(Ii, cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
plt.title(r"$I_0 (r - u(r))$")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(values[0], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])



mse = np.mean((I / I.max() - values[0] / values[0].max()) ** 2)
print("Mean Square Error:", mse)

#%%
plt.figure()
plt.imshow(u[..., 0], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
plt.colorbar(label=r'$u_x$')
plt.title(r'Componente $u_x$ del desplazamiento')

plt.figure()
plt.imshow(u[..., 1], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
plt.colorbar(label=r'$u_y$')
plt.title(r'Componente $u_y$ del desplazamiento')

plt.show()