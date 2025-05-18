import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skimage import measure
from pydata.stream_surface import PsiSolver

solver = PsiSolver(a=100, c_index=1)
r = np.linspace(0.01, 10, 200)
y = np.linspace(-1, 5, 200)
R, Y = np.meshgrid(r, y)

psi = solver.psi(R, Y)
Z = np.real(psi)

level = 16
contornos = measure.find_contours(Z, level=level)

# Convertir índice de contorno a coordenadas físicas (r, y)
contorno_escogido = max(contornos, key=len)  # usar el más largo
i_coords, j_coords = contorno_escogido[:, 0], contorno_escogido[:, 1]

# Interpolar a coordenadas reales
r_contour = np.interp(j_coords, np.arange(len(r)), r)
y_contour = np.interp(i_coords, np.arange(len(y)), y)

plt.figure()
plt.plot(r_contour, y_contour, 'k.-')
plt.xlabel("r")
plt.ylabel("y")
plt.title(r"Surface level $Re[\psi(r,y)]$ = " + f"{level}")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid(linestyle='--', alpha=0.5)
#%%
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, 'surfaces')
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
np.save("level16_200px_c2",Z)


