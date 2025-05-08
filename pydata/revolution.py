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
def export_surface_and_ring_to_obj(X, Y, Z, x_ring, y_ring, z_ring, filename):
    nrows, ncols = X.shape
    
    with open(filename, 'w') as f:
        # --- VÉRTICES superficie ---
        for i in range(nrows):
            for j in range(ncols):
                f.write(f"v {X[i, j]} {Y[i, j]} {Z[i, j]}\n")
        
        # --- CARAS superficie ---
        for i in range(nrows - 1):
            for j in range(ncols - 1):
                v1 = i * ncols + j + 1
                v2 = v1 + 1
                v3 = v1 + ncols
                v4 = v3 + 1
                f.write(f"f {v1} {v3} {v4} {v2}\n")

        # --- VÉRTICES anillo ---
        offset = nrows * ncols
        for x, y, z in zip(x_ring, y_ring, z_ring):
            f.write(f"v {x} {y} {z}\n")

        # --- LÍNEA anillo cerrada ---
        f.write("l ")
        for i in range(len(x_ring)):
            f.write(f"{offset + i + 1} ")
        f.write(f"{offset + 1}\n")  # cerrar el anillo

# Exportar
obj_path = os.path.join(base_dir, "surface_and_ring.obj")
export_surface_and_ring_to_obj(X, Y, Z3D, x_ring, y_ring, z_ring_array, obj_path)
print(f"Exportado a {obj_path}")

