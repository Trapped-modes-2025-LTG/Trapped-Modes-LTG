import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyfcd.fcd_path import compute_height_map
from pyfcd.height_map import HeightMap

base_dir = os.path.dirname(__file__)
reference_path = os.path.join(base_dir, 'Pictures', 'reference_2.png')
displaced_path = os.path.join(base_dir, 'Pictures', '202406_1457001661.bmp')
height_map = compute_height_map(reference_path, displaced_path, square_size=0.0022, height=0.0323625)       #tuple = (height_map, phases, calibration_factor)

HeightMap(*height_map).show()       #TODO: modificar para prescindir de esta clase. Desde 'height_map' se podría hacer todo
