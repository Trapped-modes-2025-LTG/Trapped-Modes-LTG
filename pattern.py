'''
Script for generate the patterns which we used for measure. 

In our setup, we place the printed pattern on top of an LED backlight, and both are positioned beneath a water tank. A camera is mounted vertically 
above the setup to capture images of the displaced pattern. Using the "pyfcd" library, we obtain the height map of the surface.

Checkerboard patterns are of particular interest. We generate them with 16-bit resolution to match the 10-bit acquisition capability of our camera.
To generate these patterns, we need to know the printer's resolution, the size of the LED backlight, and the desired size of each square in the checkerboard.

Parameters:
    N: pixels of the pattern. Deppend of the printer's resolution
    l: lenght of the LED backlight in cm. 
    mm: lenght of each square in mm
'''

import numpy as np
import tifffile
cm_i = 2.54                     # cm per inch

N = 4096                         
l = 44                          
mm = 2                          
n = l*10/mm                     # squares per lenght
dpi = N * cm_i / l              # resolution

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
