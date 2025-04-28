import numpy as np
import tifffile
cm_i = 2.54                     # cm per inch

N = 4096                        # pixels 
l = 44                          # total lenght in cm
mm = 2                          # square lenght
n = l*10/mm                     # squares per lenght
dpi = N * cm_i / l 

x = np.linspace(0, N, N)
y = np.linspace(0, N, N)
X, Y = np.meshgrid(x, y)
k = 2 * np.pi * n / N   

I0 = (((np.cos(k * X) > 0) ^ (np.cos(k * Y) > 0))).astype(float)
I0_16bit = (I0 * 65535).astype(np.uint16)

tifffile.imwrite(
    'patron.tiff',
    I0_16bit,
    photometric='minisblack',
    resolution=(dpi, dpi),
    resolutionunit='inch'  
)
