import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyfcd.fcd_new import fcd
# from pyfcd.fcd import compute_height_map
import matplotlib.pyplot as plt
from pydata.analyze import analyze

base_dir = os.path.dirname(__file__)
reference_path = os.path.join(base_dir, 'Pictures', 'reference_2.png')
displaced_path = os.path.join(base_dir, 'Pictures', '202406_1457001661.bmp')

# TODO: esto es, en orden desde abajo hasta la cámara, altura e índice del medio
layers = [[5.7e-2,1.0003], [ 1.2e-2,1.48899], [4.3e-2,1.34], [ 80e-2 ,1.0003]]

square_size=0.0022

values = fcd.compute_height_map(analyze.load_image(reference_path), analyze.load_image(displaced_path), square_size,layers)       #tuple = (height_map, phases, calibration_factor)

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
