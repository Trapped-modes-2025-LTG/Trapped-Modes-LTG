import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skimage import measure

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
file_path = os.path.join(base_dir, 'surfaces', "level16_200px_c2.npy")

Z = np.load(file_path)
level = 16
contornos = measure.find_contours(Z, level=level)

# Convertir índice de contorno a coordenadas físicas (r, y)
contorno_escogido = max(contornos, key=len)  # usar el más largo
i_coords, j_coords = contorno_escogido[:, 0], contorno_escogido[:, 1]
r = np.linspace(0.01, 10, 100)
y = np.linspace(-1, 5, 100)
# Interpolar a coordenadas reales
r_contour = np.interp(j_coords, np.arange(len(r)), r)
y_contour = np.interp(i_coords, np.arange(len(y)), y)

theta = np.linspace(0, 2 * np.pi, 200) 
Theta, R_cont = np.meshgrid(theta, r_contour)
_,     Y_cont = np.meshgrid(theta, y_contour)

X = 1 * R_cont * np.cos(Theta)
Y = R_cont * np.sin(Theta)
Z3D = Y_cont  

r_ring = 9
z_ring = 0
theta = np.linspace(0, 2 * np.pi, 200)
x_ring = r_ring * np.cos(theta)
y_ring = r_ring * np.sin(theta)
z_ring_array = np.full_like(theta, z_ring)




fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z3D, cmap='viridis', edgecolor='none')
ax.plot3D(x_ring, y_ring, z_ring_array, 'r', linewidth=2, label="Anillo r=9, z=0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
# ax.set_xlim(4,-4)
# ax.set_ylim(4,-4)
ax.set_zlim(8,-8)

#plt.savefig("superficie_3d_c2.pdf")

#%%
def export_two_surfaces_to_obj(X1, Y1, Z1, X2, Y2, Z2, filename):
    """
    Exporta dos superficies al mismo archivo .obj.
    """
    nrows1, ncols1 = X1.shape
    nrows2, ncols2 = X2.shape
    
    with open(filename, 'w') as f:
        # --- VÉRTICES superficie 1 ---
        for i in range(nrows1):
            for j in range(ncols1):
                f.write(f"v {X1[i, j]} {Y1[i, j]} {Z1[i, j]}\n")
        
        # --- VÉRTICES superficie 2 ---
        for i in range(nrows2):
            for j in range(ncols2):
                f.write(f"v {X2[i, j]} {Y2[i, j]} {Z2[i, j]}\n")

        # --- CARAS superficie 1 ---
        for i in range(nrows1 - 1):
            for j in range(ncols1 - 1):
                v1 = i * ncols1 + j + 1
                v2 = v1 + 1
                v3 = v1 + ncols1
                v4 = v3 + 1
                f.write(f"f {v1} {v3} {v4} {v2}\n")

        # --- CARAS superficie 2 ---
        offset = nrows1 * ncols1
        for i in range(nrows2 - 1):
            for j in range(ncols2 - 1):
                v1 = offset + i * ncols2 + j + 1
                v2 = v1 + 1
                v3 = v1 + ncols2
                v4 = v3 + 1
                f.write(f"f {v1} {v3} {v4} {v2}\n")

# Crear la segunda superficie (plano en z=0)
Z_plane = np.zeros_like(Z3D)

# Exportar
obj_path = os.path.join(base_dir, "surface_and_ring.obj")
export_two_surfaces_to_obj(X, Y, Z3D, X, Y, Z_plane, obj_path)
print(f"Exportado a {obj_path}")
