import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import pandas as pd
import matplotlib.pyplot as plt
from pyfcd.fcd import fcd

base_dir = os.path.dirname(__file__)
tif_folder = os.path.join(base_dir, "Pictures", "mask")
tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith('.tif')]

df_tif = pd.DataFrame({
    'file_name': tif_files,
    'file_path': [os.path.join(tif_folder, f) for f in tif_files]
})

path = df_tif["file_path"].iloc[5]      # choose one

reference_path = os.path.join(base_dir, "Pictures", "reference_df.tif")
reference = analyze.load_image(reference_path)

image = analyze.load_image(path)

mask = analyze.mask(image,
                    smoothed = 15, 
                    show_mask = True
                    )                   # 50.3 ms ± 824 μs per loop

center = analyze.center(mask)           # 21.5 ms ± 91.8 μs per loop

# Plot center
plt.scatter(center[0], center[1], s = 30, marker = "s", c = "r")

# Mesh for FCD

layers = [[5.7e-2,1.0003], [ 1.2e-2,1.48899], [4.3e-2,1.34], [ 80e-2 ,1.0003]]
square_size=0.0022

img_result = image.copy()
img_result[mask] = reference[mask]
image_to_use = img_result

height_map, _, calibration_factor = fcd.compute_height_map(
    reference,
    image_to_use,
    square_size,
    layers
)
height_map *= (1-mask)

fig, ax = plt.subplots(1,2, figsize = (9,4) )

ax[0].imshow(image_to_use)
ax[0].axis("off")
ax[0].set_title("Image to use in FCD")

im = ax[1].imshow(
    height_map,
    extent=[
        0, image_to_use.shape[1] * calibration_factor,
        0, image_to_use.shape[0] * calibration_factor 
    ]
)
ax[1].set_title("Height Map with ask")
ax[1].set_xlabel("x (m)")
ax[1].set_ylabel("y (m)")
fig.colorbar(im, ax=ax[1], label="h (m)")



