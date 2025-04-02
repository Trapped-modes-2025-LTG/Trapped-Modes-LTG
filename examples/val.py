import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyfcd.fcd import compute_height_map


def h_grad(h,x,y,k=0):
    X, Y = np.meshgrid(x, y, indexing='ij')
    N = len(X)   #asumo cuadrada
    if k == 0:
        h_hat = np.fft.fft2(h)
        Lx = np.max(X)
        Ly = np.max(Y)
        kx = (2 * np.pi / Lx) * np.fft.fftfreq(N, d=Lx/N)
        ky = (2 * np.pi / Ly) * np.fft.fftfreq(N, d=Ly/N)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        h_x_hat = 1j * KX * h_hat  
        h_y_hat = 1j * KY * h_hat  

        h_x = np.fft.ifft2(h_x_hat).real
        h_y = np.fft.ifft2(h_y_hat).real
        grad = np.stack((h_x, h_y), axis=-1)
    elif k ==1: 
        h_x, h_y = np.gradient(h, x, y, axis=(0, 1))
        grad = np.stack((h_x, h_y), axis=-1)
    else: 
        raise ValueError('Parámetro k entre 0 (pseudoespectral) y 1 (diferencias finitas)')
    return grad

def val(func,k, N = 500, Lx = np.pi, Ly = np.pi, H = 1, square_size = 1, kx = 20, ky = 20, centrado_si = False, *args, **kwargs):
    '''
    Parámetros necesarios:
        'func': función usada. Necesariamente tiene que ser función de X e Y, con estructura 'func(X,Y,*kwargs*)'
        'k': integrador a usar. k = 0 pseudoespectral, k = 1 diferencias finitas
        '**kwargs': parámetros necesarios de 'func', como amplitud, fase, frecuencia, o lo necesario.
    Parámetros opcionales (llamarlos con otro valor si se quiere modificarlos):
        'N = 500': Grillado X e Y
        'Lx = Ly = np.pi': puntos finales de ambas coordenadas
        'H = 1': altura efectiva del agua. 
        'kx = ky = 20': frecuencia del patrón I_0
        'centrado_si = False': Centrar y normalizar los valores devueltos por la FCD
    '''
    x = np.linspace(0, Lx, N)
    y = np.linspace(0, Ly, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    h = func(X,Y,*args, **kwargs)
    u = -H*h_grad(h,x,y,k=k)
    I0 = np.sin(kx * X) * np.sin(ky * Y)
    r = np.stack((X, Y), axis=-1)
    r_prim = (r - u)  

    interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
    I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N) 
    Ii = I.T
    Ii = np.flip(Ii, axis=(0, 1))  # Invierte en ambas direcciones
    values = compute_height_map(I0, Ii, square_size=square_size, height=H)[0]     #tuple = (height_map, phases, calibration_factor)
    
    if centrado_si == True:
        values = centrado(values)
    else:
        pass
    
    return X,Y,h, Ii, values, I0

def centrado(v):
    meanh = (np.max(v) + np.min(v))*0.5
    difh = (np.max(v) - np.min(v))*0.5
    return (v-meanh)/difh
