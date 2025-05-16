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

#%% Para los toroides
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

X = R_cont * np.cos(Theta)
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

#%% Veo las derivadas para orientarme, de todos modos elijo cortar mucho mas adelante, para no forzar a la impresora
# puede ser un problema porque no hay tanto juego con la superficie libre

y_corte = 0.4 # mas o menos es el -9

dy_dr = np.gradient(y_contour, r_contour)
sign_change_indices = np.where(np.diff(np.sign(dy_dr)))[0]
idx_inicio = sign_change_indices[2]

print(idx_inicio)
idx_inicio = 35
r_recortado = r_contour[idx_inicio:-8]
y_recortado = y_contour[idx_inicio:-8]

r0 = r_recortado[0]
y0 = y_recortado[0]

y1 = -2
r1 = r0

n_base = 20
y_b = np.full(n_base, -2.5)
r_b = np.linspace(r_recortado[-1]-0.0, 7, n_base)

np1 = 10
np2 = 10
r_linea1 = np.full(np1, r0)
y_linea1= np.linspace(-1.1, y0, np1)
r_linea2 = np.full(np2, r_recortado[-1])
y_linea2= np.linspace(y_recortado[-1], -1.66, np2)

y_conc = np.concatenate([y_linea1[:-1],y_recortado[:-1],y_linea2])
r_conc = np.concatenate([r_linea1[:-1],r_recortado[:-1],r_linea2])

plt.figure() 
plt.plot(r_contour, y_contour, label = 'Original')
plt.plot(r_conc, y_conc, 'k.-', label = 'Flotante')
plt.scatter(5.5201, 0, marker = 'o', color = 'r', label = 'Fuente')
plt.plot(r_b, y_b, 'k-', label = 'Soporte')
plt.legend(fontsize = 12, loc = 'lower right', ncol = 2)
plt.grid(linestyle='--', alpha=0.5)
plt.xlabel('r', fontsize = 14)
plt.ylabel('y', fontsize = 14)
# plt.savefig('flotante_modificado.pdf', bbox_inches = 'tight')

#%%
# Ya acá se ve que la interpolacion no es tan buena, la figura es re fea
# Esto hace que, cuando se le da grosor en blender, haya muchos errores
# La figura debe ser continua, entonces mas que concatenar se debería interpolar 

sort_idx = np.argsort(r_recortado)
r_conc_sorted = r_conc
y_conc_sorted = y_conc

# 2. Generar malla de revolución
theta = np.linspace(0, 2*np.pi, 100)
R, Theta = np.meshgrid(r_conc_sorted, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = np.tile(y_conc_sorted, (len(theta), 1))

# 3. Graficar
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_zlim(-8,8)
plt.show()

#%% Esto es para la base

y_b = np.full(n_base, - 2.5)
r_b = np.linspace(r_recortado[-1], 7, n_base)

Rb, Thetab = np.meshgrid(r_b, theta)
Xb = Rb * np.cos(Thetab)
Yb = Rb * np.sin(Thetab)
Zb = np.tile(y_b, (len(theta), 1))

# 3. Graficar
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xb, Yb, Zb, cmap='viridis', alpha=0.8)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_zlim(-8,8)
plt.show()


#%% Guardar y exportar
def export_surface_and_ring_to_obj(X, Y, Z, filename):
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

obj_path = os.path.join(base_dir, "base_1.obj")
export_surface_and_ring_to_obj(Xb, Yb, Zb, obj_path)
print(f"Exportado a {obj_path}")

#%%
import trimesh

# Datos de entrada (usá tus valores reales)
theta = np.linspace(0, 2*np.pi, 360)  # más suave
r = r_conc_sorted                     # debe estar en orden creciente
z = y_conc_sorted                     # eje vertical en Blender (Z)

# Crear los puntos de revolución
R, Theta = np.meshgrid(r, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = np.tile(z, (len(theta), 1))

# 1. Crear vértices
vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

# 2. Crear caras
faces = []
n_r = len(r)
n_theta = len(theta)

for i in range(n_theta - 1):
    for j in range(n_r - 1):
        a = i * n_r + j
        b = a + 1
        c = a + n_r
        d = c + 1
        faces.append([a, c, b])
        faces.append([b, c, d])


# Crear la malla
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

# Exportar a archivo OBJ
mesh.export("concatenado_2.obj")
print("Exportado como figura_revolucion.obj")

#%%
# 1. Crear vértices
vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

# 2. Crear caras
faces = []
n_r = len(r)
n_theta = len(theta)

for i in range(n_theta - 1):
    for j in range(n_r - 1):
        a = i * n_r + j
        b = a + 1
        c = a + n_r
        d = c + 1
        faces.append([a, c, b])
        faces.append([b, c, d])

# Cerrar la base (círculo inferior)
center_bottom = [0, 0, z[0]]
center_top = [0, 0, z[-1]]
vertices = np.vstack([vertices, center_bottom, center_top])
center_bottom_idx = len(vertices) - 2
center_top_idx = len(vertices) - 1

# Agregar caras del círculo inferior
for i in range(n_theta - 1):
    a = i * n_r
    b = (i + 1) * n_r
    faces.append([a, b, center_bottom_idx])

# Agregar caras del círculo superior
for i in range(n_theta - 1):
    a = i * n_r + (n_r - 1)
    b = (i + 1) * n_r + (n_r - 1)
    faces.append([b, a, center_top_idx])

# Crear la malla
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

# Exportar a archivo OBJ
mesh.export("concatenado_2.obj")

# Base hueca en z = -2.5
theta = np.linspace(0, 2*np.pi, 360)
r_base = np.linspace(r_recortado[-1], 7, 50)  # del radio interior al exterior
z_base = np.full_like(r_base, -2.5)           # constante

# Malla de revolución
R, Theta = np.meshgrid(r_base, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = np.tile(z_base, (len(theta), 1))

# Vértices
vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

# Caras
faces = []
n_r = len(r_base)
n_theta = len(theta)

for i in range(n_theta - 1):
    for j in range(n_r - 1):
        a = i * n_r + j
        b = a + 1
        c = a + n_r
        d = c + 1
        faces.append([a, c, b])
        faces.append([b, c, d])

# Malla
mesh_base = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
mesh_base.export("base_hueca_revolucion.obj")
print("Exportado como base_hueca_revolucion.obj")

#%%

C = r_recortado[-1]/4.5