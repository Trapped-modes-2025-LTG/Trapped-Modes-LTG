import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyval.val import val

def Asin(X,Y, A=1, w = 3):
    return A*np.sin(w*X)

alturas = np.linspace(0.01, 1, 20)
all_data = []
vmin, vmax = np.inf, -np.inf

for  altura in alturas: 
    X,Y, h,Ii, values,I0 = val(Asin,A = altura, centrado_si = True, k = 0)
    data = values - h.T/np.max(h.T)
    all_data.append(data)
    vmin = min(vmin, np.min(data))
    vmax = max(vmax, np.max(data))

N = 500

fila_corte = N // 2  # Índice de la fila central

cmap = cm.gnuplot  # Colormap 'magma'
norm = mcolors.Normalize(vmin=np.min(alturas), vmax=np.max(alturas))  # Normalización

fig, ax = plt.subplots()
#rec = []
for altura, data in zip(alturas, all_data):
    x_vals = np.linspace(0, np.max(X), N)  # Eje X
    y_vals = data[fila_corte, :]  # Corte en la fila central
    color = cmap(norm(altura))  # Color según altura

    ax.plot(x_vals, y_vals, color=color, label=f"H = {altura:.2f}")

# Crear la barra de color
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Altura")

ax.set_xlabel("x")
ax.set_ylabel("Intensidad")
ax.set_title("Cortes en la fila central para distintas alturas")
ax.grid(linestyle = '--', alpha = 0.5)
ax.plot()
# plt.imshow(u[..., 1], cmap="magma", origin="lower", extent=[0, Lx, 0, Ly])
# plt.colorbar(label=r'$u_y$')
# plt.title(r'Componente $u_y$ del desplazamiento')

# plt.show()
