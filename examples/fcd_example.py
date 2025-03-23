route = '/home/juan/fcd-analysis/Codigo nuevo/prueba1'
reference_path = '/home/juan/fcd-analysis/examples/static_images/reference_2.png'
#reference_path = f"{route}/prueba1_20250317_122608_C1S0001000001.tif"
displaced_path = f"{route}/prueba1_20250317_122608_C1S0001000158.tif"
displaced_path = '/home/juan/fcd-analysis/examples/propagating_waves/202406_1457001661.bmp'
# Crear el mapa de altura
height_map = compute_height_map(reference_path, displaced_path, square_size=0.0022, height=0.0323625)

height_map.show()