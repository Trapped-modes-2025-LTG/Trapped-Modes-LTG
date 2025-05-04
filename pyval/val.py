'''
This script contain the main structure to validate the FCD method [1] with synthetic data. A code of its 
implementation is located in "/examples/val_example.py".

The idea is as follows. It's possible to build the displaced pattern integrating the surface height map. For
this, we use the equation (1) and the definition of u(r) in section 4.1, both in [1]. Then, It's neccesary 
to send the displaced pattern to "compute_height_map()", which is the main function of the method, for rebuild 
the height map. Validation consist of comparate both height maps.

A synthetic height map is built using a provided function. Its integral is used as the displacement field, 
which is then used to interpolate the original pattern and construct the synthetic image. 

There are two things that cannot be validated with this script. The first one is the effective heihgt of the 
surface: the FCD code used differents layers to determinate it [2]. The second one is the "calibration_factor",
the code were built to set it as 1, setting the correct "square_size". This was because the synthetic height map 
didn't have physical units. 

Other libraries of the repository are needed. Check its correct call.

[1] Wildeman, Sander. “Real-Time Quantitative Schlieren Imaging by Fast Fourier Demodulation of a Checkered 
Backdrop.” Experiments in Fluids 59, no. 6 (June 2018): 97. https://doi.org/10.1007/s00348-018-2553-9.
[2] Moisy, Frédéric, Marc Rabaud, and Kévin Salsac. “A Synthetic Schlieren Method for the Measurement of the 
Topography of a Liquid Interface.” Experiments in Fluids 46, no. 6 (June 2009): 1021–36. 
https://doi.org/10.1007/s00348-008-0608-z.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyfcd.fcd import compute_height_map
from pyfcd.fourier_space import wavenumber_meshgrid

def val(k,func = None,h = None, N = 1024,  H = 1, n = 60, centrado_si = False, *args, **kwargs):
    '''
    Parameters:
        "k" integer (0 or 1).- Select the integrator to use. 
            k = 0 for pseudospectral
            k = 1 for finite differences
        "func = None": function.- used function. It needs to be funcion of X and Y, with an estructure as 'func(X,Y,*kwargs*)'.
        "h = None": array.- Used if we want to validate the method with data. If h isn't None, func must be None.
        "N = 1024": integer.- Axis's resolution.
        "H = 1": integer.- Water's effective height. 
        "n = 60": integer.- Light's squares per lenght.
        "centrado_si = False": boolean.- Center and normalyze the return values of the FCD. 
            Try to avoid this parameter. It's neccesary when the mean of the funcion used is not zero.  
        "**kwargs": list.- Parameters of 'func', as amplitude, phase, frequency, or whatever.
    '''
    
    square_size = N/(2*n)       # for calibration_factor = 1
    
    x = np.linspace(0, N, N, endpoint = False)
    y = np.linspace(0, N, N, endpoint = False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    if func is not None:
        if h is None:
            h = func(X,Y,*args, **kwargs)
        else:
            raise Warning("Provide either h or func, not both.")
    else: 
        if h is None:
            raise Warning("Provide either h or func, not empty.")
        else: 
            h = h
    u = -H*h_grad(h,x,y,k=k)
    
    kx = 2 * np.pi * n / N
    ky = 2 * np.pi * n / N
    I0 = 0.5 + (np.sin(X *kx) * np.sin(Y * ky))/2
    r = np.stack((X, Y), axis=-1)
    r_prim = (r - u)  
    r_prim[..., 0] = np.clip(r_prim[..., 0], x.min(), x.max())
    r_prim[..., 1] = np.clip(r_prim[..., 1], y.min(), y.max())

    interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
    I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N) 
    #I = 0.5 + (np.cos(r_prim[..., 0] * kx) * np.cos(r_prim[..., 1] * ky)) / 2    #interpolo directamente evaluando en el patron
    values = compute_height_map(I0, I, square_size=square_size, height=H)    #tuple = (height_map, phases, calibration_factor)
    calibration_factor = values[2]
    if centrado_si == True:
        values[0] = centrado(values[0])
    else:
        pass
    return X,Y,h, I, values[0], I0, calibration_factor

def h_grad(h,x,y,k=0):  
    '''
    Integrator to use.
    '''
    if k == 0:
        kx, ky = wavenumber_meshgrid(h.shape)
        h = h-np.mean(h)
        h_hat = np.fft.fft2(h)
        h_x_hat = 1j * kx * h_hat
        h_y_hat = 1j * ky * h_hat
        h_x = np.fft.ifft2(h_x_hat).real
        h_y = np.fft.ifft2(h_y_hat).real
        grad = np.stack((h_x, h_y), axis=-1)
    elif k ==1: 
        h_x, h_y = np.gradient(h, x, y, axis=(0, 1))
        grad = np.stack((h_x, h_y), axis=-1)
    else: 
        raise ValueError('Parameter k must be an integer between 0 (pseudoespectral) and 1 (finite differences)')
    return grad

def centrado(v):
    meanh = (np.max(v) + np.min(v))*0.5
    difh = (np.max(v) - np.min(v))*0.5
    return (v-meanh)/difh
