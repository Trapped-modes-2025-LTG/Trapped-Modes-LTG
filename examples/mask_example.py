import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import pandas as pd
import matplotlib.pyplot as plt
from pyfcd.fcd import fcd
import numpy as np

base_dir = os.path.dirname(__file__)
#%%
tif_folder = os.path.join(base_dir, "Pictures", "mask")

tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith('.tif')]

df_tif = pd.DataFrame({
    'file_name': tif_files,
    'file_path': [os.path.join(tif_folder, f) for f in tif_files]
})

path = df_tif["file_path"].iloc[5]      # choose one
image = analyze.load_image(path)

mask = analyze.mask(image,
                    smoothed = 10, 
                    percentage = 90,
                    show_mask = True
                    )
#%%

base_dir2 = os.path.abspath(os.path.join(base_dir, '..'))

folder_29_5 = os.path.join(base_dir2, "datos", "29_5", "prueba")

tif_files2 = [f for f in os.listdir(folder_29_5) if f.lower().endswith('.tif')]

df_tif2 = pd.DataFrame({
    'file_name': tif_files2,
    'file_path': [os.path.join(folder_29_5, f) for f in tif_files2]
})

path = df_tif2["file_path"].iloc[5]      # choose one
image = analyze.load_image(path)

mask, contornos = analyze.mask(image,
                    smoothed = 15, 
                    percentage = 95,
                    show_mask = False
                    )

contorno = contornos[0]  # Suponiendo que es uno solo
p1 = contorno[0]
p2 = contorno[-1]

y_max = mask.shape[0] - 1  # 1023

y1, x1 = p1.astype(int)
y2, x2 = p2.astype(int)

camino1_x = np.linspace(x1, y_max, int(y_max-x1))

camino2_y = np.linspace(0,y2,int(y2)) 

if 0 in camino1_x:
    camino1_y = np.full_like(camino1_x, 1023)
elif 1023 in camino1_x:
    camino1_y = np.zeros_like(camino1_x)
    
if 0 in camino2_y:
    camino2_x = np.full_like(camino2_y, 1023)  
elif 1023 in camino2_y:
    camino2_x = np.zeros_like(camino2_y)
    
    
displaced = mask.T*image

reference_path = os.path.join(base_dir2, "datos", "29_5", "reference.tif")

reference = analyze.load_image(reference_path)

# plt.figure()
# plt.imshow(displaced)
# for c in contornos:
#     plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')

layers = [[5.7e-2,1.0003], [ 1.2e-2,1.48899], [3.4e-2,1.34], [ 80e-2 ,1.0003]]

square_size = 0.002

displaced = np.where((mask.T),displaced, reference)

plt.figure()
plt.imshow(displaced-reference)

values = fcd.compute_height_map(reference, displaced, square_size, layers)
plt.figure()
plt.imshow(values[0])
plt.colorbar()

