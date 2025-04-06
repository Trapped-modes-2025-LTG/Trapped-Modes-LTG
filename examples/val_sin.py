import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyval.val import val

def Asin(X,Y, A=1, w = 3):
    return A*np.sin(w*X)
def diag_sin(X,Y, A=1, w = 3):
    return A*np.sin(w*(X+Y))
#%%
X,Y, h,Ii, values,I0 = val(Asin, k = 0,A = 0.1,kx = 80, ky = 80, centrado_si = False)
# plt.figure()
# plt.title('original')
# plt.imshow(values, origin = 'lower')
# plt.colorbar()
# plt.figure()
# plt.title('FCD')
# plt.imshow(h, origin = 'lower')
# plt.colorbar()
#%%
alturas = np.linspace(0.01,0.4, 20)
all_data = []
vmin, vmax = np.inf, -np.inf

N = 500

for  altura in alturas: 
    X,Y, h,Ii, values,I0 = val(Asin, k = 0,A = altura,kx = 100, ky = 100, centrado_si = True)
    data = values - h.T/np.max(h.T)
    all_data.append(data)
    vmin = min(vmin, np.min(data))
    vmax = max(vmax, np.max(data))

fila_corte = N // 2  # Índice de la fila central

cmap = cm.gnuplot 
norm = mcolors.Normalize(vmin=np.min(alturas), vmax=np.max(alturas))  # Normalización

fig, ax = plt.subplots()
for altura, data in zip(alturas, all_data):
    x_vals = np.linspace(0, np.max(X), N)  # Eje X
    y_vals = data[fila_corte, :]  # Corte en la fila central
    color = cmap(norm(altura))  # Color según altura

    ax.plot(x_vals, y_vals, color=color, label=f"H = {altura:.2f}")

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Altura")

ax.set_xlabel("x")
ax.set_ylabel("Resta alturas")
ax.set_title("Cortes en la fila central para distintas alturas")
ax.grid(linestyle = '--', alpha = 0.5)
ax.plot()

#%%
all_data = []
vmin, vmax = np.inf, -np.inf
ks = np.linspace(40,100,7)

for  k in ks: 
    X,Y, h,Ii, values,I0 = val(Asin, k = 0,A = 0.1,kx = k, ky = k, centrado_si = True)
    data = values - h.T/np.max(h.T)
    all_data.append(data)
    vmin = min(vmin, np.min(data))
    vmax = max(vmax, np.max(data))

fila_corte = N // 2  # Índice de la fila central

cmap = cm.gnuplot 
norm = mcolors.Normalize(vmin=np.min(ks), vmax=np.max(ks))  # Normalización

fig, ax = plt.subplots()
for k, data in zip(ks, all_data):
    x_vals = np.linspace(0, np.max(X), N)  # Eje X
    y_vals = data[fila_corte, :]  # Corte en la fila central
    color = cmap(norm(k))  # Color según altura
    ax.plot(x_vals, y_vals, color=color)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Frecuencia patrón")

ax.set_xlabel("x")
ax.set_ylabel("Resta alturas")
ax.set_title("Cortes en la fila central para \n diferentes frecuencias y A = 0.1")
ax.grid(linestyle = '--', alpha = 0.5)
plt.savefig('sim_sin_difk.pdf', bbox_inches = 'tight')

