'''
This script contain the main structure to validate the FCD method [1] with synthetic data. An implementation 
example can be found in "/examples/val_example.py".

The validation procedure is as follows: the displaced pattern is constructed by integrating a 
synthetic surface height map. This is done using equation (1) and the definition of u(r)  from Section 4.1 
of [1]. The resulting displaced pattern is then processed with the  "compute_height_map()" function, which 
is the core of the FCD method, to reconstruct the height map. Validation consists in comparing the original 
and reconstructed height maps.

A synthetic height map is generated using a user-provided function. Its integral is used as a displacement 
field to interpolate the original pattern and create the synthetic image.

There are two aspects that cannot be validated with this script:
1. The effective height of the surface: the FCD implementation uses multiple optical layers to estimate 
    this [2].
2. The calibration factor: the code sets this to 1 by choosing an appropriate "square_size". This is because 
    the synthetic height map does not contain physical units.

Make sure all required libraries from the repository are correctly imported.

Rerences:
[1] Wildeman, Sander. “Real-Time Quantitative Schlieren Imaging by Fast Fourier Demodulation 
    of a Checkered Backdrop.” *Experiments in Fluids*, vol. 59, no. 6, June 2018, p. 97. 
    https://doi.org/10.1007/s00348-018-2553-9
[2] Moisy, Frédéric, Marc Rabaud, and Kévin Salsac. “A Synthetic Schlieren Method for the Measurement 
    of the Topography of a Liquid Interface.” *Experiments in Fluids*, vol. 46, no. 6, June 2009, 
    pp. 1021–36. https://doi.org/10.1007/s00348-008-0608-z
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
        k : int (0 or 1)
            Selects the integration method.
                0: Pseudospectral method
                1: Finite difference method

        func : callable, optional
            Function used to generate the synthetic height map. 
            Must accept (X, Y, *args, **kwargs) as input.

        h : ndarray, optional
            Precomputed height map. If provided, "func" must be None.

        N : int, default=1024
            Resolution of the domain along each axis.

        H : float, default=1
            Effective height of the medium (e.g., water) used in the displacement calculation.

        n : int, default=60
            Number of squares per image width in the checkerboard pattern.

        centrado_si : bool, default=False
            Whether to normalize the reconstructed height map to mean 0 and max amplitude 1. 
            Useful if the original height map does not have zero mean.

        *args, **kwargs:
            Additional arguments passed to "func".

    Returns:
        tuple of ndarrays:
            X, Y         : meshgrid coordinates
            h            : original height map
            I            : distorted image pattern
            values[0]    : reconstructed height map
            I0           : original undistorted pattern
            calibration_factor : scaling factor estimated by FCD (should be 1 in this setup)
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
    #I = 0.5 + (np.cos(r_prim[..., 0] * kx) * np.cos(r_prim[..., 1] * ky)) / 2    # Direct interpolate if the pattern is known
    values = compute_height_map(I0, I, square_size=square_size, height=H)    # Return: tuple = (height_map, phases, calibration_factor)
    calibration_factor = values[2]
    if centrado_si == True:
        values[0] = centrado(values[0])
    else:
        pass
    return X,Y,h, I, values[0], I0, calibration_factor

def h_grad(h,x,y,k=0):  
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
