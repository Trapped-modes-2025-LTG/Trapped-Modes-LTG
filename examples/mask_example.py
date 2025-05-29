import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import pandas as pd

base_dir = os.path.dirname(__file__)

tif_folder = os.path.join(base_dir, "Pictures", "mask")

tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith('.tif')]

df_tif = pd.DataFrame({
    'file_name': tif_files,
    'file_path': [os.path.join(tif_folder, f) for f in tif_files]
})

path = df_tif["file_path"].iloc[5]      # choose one
image = analyze.load_image(path)

mask = analyze.mask(image,
                    smoothed = 15, 
                    percentage = 90,
                    show_mask = True
                    )