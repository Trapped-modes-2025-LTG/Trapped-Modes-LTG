import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze

base_dir = os.path.dirname(__file__)

tif_folder = os.path.join(base_dir, "Pictures", "mask")

reference_path = os.path.join(base_dir, "Pictures", "reference_df.tif")

layers = [[3.2e-2,1.0003], [ 1.2e-2,1.48899], [3.4e-2,1.34], [ 80e-2 ,1.0003]]

analyze.folder(reference_path, tif_folder, layers, 0.002, smoothed=15, percentage =95 , timer= False)

