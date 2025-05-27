import numpy as np
import tifffile

# Constantes
cm_i = 2.54   # cm por pulgada
mm = 1     # tamaño de cada cuadrado
cuadros_x = 50 * 10 // mm   # 250 cuadrados
cuadros_y = 84 * 10 // mm   # 420 cuadrados

pixeles_por_cuadro = 10     # resolución deseada → 20 px por 2 mm = 100 px/cm

# Tamaño total de la imagen
Nx = cuadros_x * pixeles_por_cuadro  # 250*20 = 5000
Ny = cuadros_y * pixeles_por_cuadro  # 420*20 = 8400

# Crear patrón de tablero de ajedrez
tile = np.array([[0, 1], [1, 0]], dtype=np.uint8)
checkerboard = np.tile(tile, (cuadros_y//2, cuadros_x//2))

# Escalamos cada cuadro a pixeles
I0 = np.kron(checkerboard, np.ones((pixeles_por_cuadro, pixeles_por_cuadro)))

# Convertimos a 16 bits
I0_16bit = (I0 * 65535).astype(np.uint16)

# Resolución en DPI
dpi = pixeles_por_cuadro * (10 / mm) * cm_i  # = 100 px/cm * 2.54 = 254 dpi

# Guardar imagen
tifffile.imwrite(
    'patron_1_5mm.tiff',
    I0_16bit,
    photometric='minisblack',
    resolution=(dpi, dpi),
    resolutionunit='inch'
)
