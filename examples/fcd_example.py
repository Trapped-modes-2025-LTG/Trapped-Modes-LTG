import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyfcd.fcd import compute_height_map
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)
reference_path = os.path.join(base_dir, 'Pictures', 'reference_2.png')
displaced_path = os.path.join(base_dir, 'Pictures', '202406_1457001661.bmp')
values = compute_height_map(reference_path, displaced_path, square_size=0.0022, height=0.0323625)       #tuple = (height_map, phases, calibration_factor)

x = np.linspace(0,values[0].shape[1],values[0].shape[1]) * values[2]
y = np.linspace(0,values[0].shape[0],values[0].shape[0]) * values[2]
x_mesh, y_mesh = np.meshgrid(x,y)

fig, axis = plt.subplots()        #Heightmap
im = axis.contourf(x_mesh * 1e3, y_mesh * 1e3, values[0] * 1e3, 100)    
cbar = fig.colorbar(im, ax=axis)
cbar.set_label('Altura [mm]', rotation=270, labelpad=15)
axis.set_xlabel('Posición x [mm]')
axis.set_ylabel('Posición y [mm]')
axis.set_aspect("equal")    

fig, axs = plt.subplots(1, 2)    #phases
for i, angles in enumerate(values[1]):
    im = axs[i].contourf(x_mesh*1e3, y_mesh*1e3, values[1][i], 100)
    for c in im.collections: # Esto soluciona el problema de aliasing al guardar como .pdf.
        c.set_edgecolor("face")
    cbar = fig.colorbar(im, ax=axs[i])
    cbar.set_label('Ángulo [rad]', rotation=270, labelpad=15)
    axs[i].set_xlabel('Posición x [mm]')
    axs[i].set_ylabel('Posición y [mm]')
    axs[i].set_aspect("equal")
    axs[i].set_title(f"Phi_{i+1}.")
plt.tight_layout()
plt.show()
