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

displaced = mask.T*image

reference_path = os.path.join(base_dir2, "datos", "29_5", "reference.tif")

reference = analyze.load_image(reference_path)

plt.figure()
plt.imshow(reference-displaced)

layers = [[5.7e-2,1.0003], [ 1.2e-2,1.48899], [3.4e-2,1.34], [ 80e-2 ,1.0003]]

square_size = 0.002

displaced = np.where((mask),displaced, reference)

values = fcd.compute_height_map(reference, displaced, square_size,layers)

plt.figure()
plt.imshow(values[0])

# #%%
# import cv2
# import numpy as np

# cnt1 = contornos[0].astype(np.int32)
# cnt2 = contornos[1].astype(np.int32)

# # Asegurarse de que la forma sea 2D
# mask_shape = image.shape[:2]  # o usa mask.reshape((H, W)) si es necesario
# outer_mask = np.zeros(mask_shape, dtype=np.uint8)
# inner_mask = np.zeros(mask_shape, dtype=np.uint8)

# cv2.drawContours(outer_mask, [cnt1], -1, color=1, thickness=cv2.FILLED)
# cv2.drawContours(inner_mask, [cnt2], -1, color=1, thickness=cv2.FILLED)

# between_mask = outer_mask - inner_mask
# between_mask[between_mask < 0] = 0
# mask = 1-between_mask
# plt.figure()
# plt.imshow(mask, cmap='gray')
# plt.title("Máscara entre contornos")
# plt.axis('off')
# plt.show()

#%%