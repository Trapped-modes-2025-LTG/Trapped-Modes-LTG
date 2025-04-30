import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyval.val import val

def step(X,a= 0.02,w = 400):
    x0 = len(X)//2
    return 1 / (1 + np.exp(-a * (X - x0 + w/2))) * 1 / (1 + np.exp(a * (X - x0 - w/2)))
def gauss_sin(X,Y, A = 100, w = 0.05):
    return step(X)*step(Y)*A*np.sin(w*(X+Y))
    
# elegir la función deseada, y si se quiere cambiar los parámetros:
    # el integrador (k = 0 o k = 1)
    # la frecuencia de puntos N
    # la altura efectiva H
    # el espaciado de los puntos del patrón n
# en este caso, está todo definido para que el factor de calibración devuelto sea 1. Simplemente para la simulación.
    
X,Y, h,Ii, values,I0, calibration_factor = val(0, func = gauss_sin, centrado_si = False)

fig, ax = plt.subplots(1,3, figsize = (10,4))

im0 = ax[0].imshow(Ii, origin='lower')
cbar = fig.colorbar(im0, ax=ax[0], orientation='horizontal')
ax[0].set_title('Patrón deformado')

im1 = ax[1].imshow(values, origin='lower')
cbar = fig.colorbar(im1, ax=ax[1], orientation='horizontal')
ax[1].set_title('Mapa devuelto por la FCD')

im2 = ax[2].imshow(values-h, origin='lower')
cbar = fig.colorbar(im2, ax=ax[2], orientation='horizontal')
ax[2].set_title(r'Resta de alturas: <' + f'{np.max(np.abs(values-h))*100/(np.max(np.abs(values))):.2f}%')
