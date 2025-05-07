import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skimage import measure

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
file_path = os.path.join(base_dir, 'surfaces', "level16_200px.npy")

Z = np.load(file_path)
level = 16
contornos = measure.find_contours(Z, level=level)

# Convertir índice de contorno a coordenadas físicas (r, y)
contorno_escogido = max(contornos, key=len)  # usar el más largo
i_coords, j_coords = contorno_escogido[:, 0], contorno_escogido[:, 1]
r = np.linspace(0.01, 3, 200)
y = np.linspace(-1, 1.1, 200)
# Interpolar a coordenadas reales
r_contour = np.interp(j_coords, np.arange(len(r)), r)
y_contour = np.interp(i_coords, np.arange(len(y)), y)

theta = np.linspace(0, 2 * np.pi, 200) 
Theta, R_cont = np.meshgrid(theta, r_contour)
_,     Y_cont = np.meshgrid(theta, y_contour)

X = 1.2 * R_cont * np.cos(Theta)
Y = R_cont * np.sin(Theta)
Z3D = Y_cont  

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z3D, cmap='viridis', edgecolor='none')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(4,-4)
ax.set_ylim(4,-4)
ax.set_zlim(3,-3)