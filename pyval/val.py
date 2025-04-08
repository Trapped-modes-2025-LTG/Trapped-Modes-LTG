import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyfcd.fcd import compute_height_map
from pyfcd.fourier_space import wavenumber_meshgrid

def h_grad(h,x,y,k=0):
    X, Y = np.meshgrid(x, y, indexing='ij')    
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
        raise ValueError('Parámetro k entre 0 (pseudoespectral) y 1 (diferencias finitas)')
    return grad

def val(k,func = None,h = None, N = 1024,  H = 1, square_size = 1, kx = 20, ky = 20, centrado_si = False, *args, **kwargs):
    '''
    Parámetros necesarios:
        'k': integrador a usar. k = 0 pseudoespectral, k = 1 diferencias finitas
        'func': función usada. Necesariamente tiene que ser función de X e Y, con estructura 'func(X,Y,*kwargs*)'
        'h': campo de alturas directamente de tenerlo      
        '**kwargs': parámetros necesarios de 'func', como amplitud, fase, frecuencia, o lo necesario.
    Parámetros opcionales (llamarlos con otro valor si se quiere modificarlos):
        'N = 1024': Grillado X e Y
        'H = 1': altura efectiva del agua. 
        'kx = ky = 20': frecuencia del patrón I_0
        'centrado_si = False': Centrar y normalizar los valores devueltos por la FCD
    '''
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
    I0 = np.sin(kx * X) * np.sin(ky * Y)
    r = np.stack((X, Y), axis=-1)
    r_prim = (r - u)  
    r_prim[..., 0] = np.clip(r_prim[..., 0], x.min(), x.max())
    r_prim[..., 1] = np.clip(r_prim[..., 1], y.min(), y.max())

    interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
    I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N) 
    values = compute_height_map(I0, I, square_size=square_size, height=H)[0]     #tuple = (height_map, phases, calibration_factor)
    
    if centrado_si == True:
        values = centrado(values)
    else:
        pass
    return X,Y,h, I, values, I0

def centrado(v):
    meanh = (np.max(v) + np.min(v))*0.5
    difh = (np.max(v) - np.min(v))*0.5
    return (v-meanh)/difh
