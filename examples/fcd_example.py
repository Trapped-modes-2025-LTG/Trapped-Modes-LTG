import os
route = '/home/juan/fcd-analysis/Codigo nuevo/prueba1'

reference_path = os.path.join('examples/pictures/', 'reference_2.png')
displaced_path = os.path.join('examples/pictures/', '202406_1457001661.bmp')
height_map = compute_height_map(reference_path, displaced_path, square_size=0.0022, height=0.0323625)

height_map.show()
