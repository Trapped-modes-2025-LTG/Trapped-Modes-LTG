import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skimage import measure

a = 100
c = 2.4048

def En(n, z):
    if n <= 0:
        raise Exception("n must be greater than 0.")
    elif n == 1:
        return sp.exp1(z)
    else:
        return (np.exp(-z) - z * En(n - 1, z)) / (n - 1)

def L(d, y, a):
    return -np.imag(En(1,(d + 1j * y) * a)) + np.real(En(1,(d + 1j * y) * a)) / a

def integrando1(nu, r, y):
    return (nu * np.sin(nu * y) + np.cos(nu * y)) * sp.iv(1, nu * r) * sp.kv(0, nu * c) * nu / (nu ** 2 + 1)

def integrando2(nu, r, y):
    return (nu * np.sin(nu * y) + np.cos(nu * y)) * sp.iv(0, nu * c) * sp.kv(1, nu * r) * nu / (nu ** 2 + 1)

def integral1(r, y):
    return quad(integrando1, 0, a, args=(r, y))[0]

def integral2(r, y):
    return quad(integrando2, 0, a, args=(r, y))[0]

def psi1(r, y):
    M = lambda r, y: r * np.exp(-y) * sp.jv(1, r) * sp.hankel1(0, c)
    return -4 * np.pi ** 2 * 1j * c * M(r, y) - 8 * c * r * integral1(r, y) + L(np.abs(r - c), y, a) / (2 * np.sqrt(r * c))

def psi2(r, y):
    M = lambda r, y: r * np.exp(-y) * sp.jv(0, c) * sp.hankel1(1, r)
    return -4 * np.pi ** 2 * 1j * c * M(r, y) + 8 * c * r * integral2(r, y) + L(np.abs(r - c), y, a) / (2 * np.sqrt(r * c))

def psi(r, y):
    sol = np.zeros(np.shape(r), dtype=complex)
    for i in range(len(r[:, 0])):
        for j in range(len(r[0, :])):
            sol[i, j] = psi1(r[i, j], y[i, j]) if (r[i, j] < c) else psi2(r[i, j], y[i, j])
    return sol

#%%

r = np.linspace(0.01, 3, 200)
y = np.linspace(-1, 1.1, 200)
R, Y = np.meshgrid(r, y)
Z = np.real(psi(R, Y))

level = 16
contornos = measure.find_contours(Z, level=level)

# Convertir índice de contorno a coordenadas físicas (r, y)
contorno_escogido = max(contornos, key=len)  # usar el más largo
i_coords, j_coords = contorno_escogido[:, 0], contorno_escogido[:, 1]

# Interpolar a coordenadas reales
r_contour = np.interp(j_coords, np.arange(len(r)), r)
y_contour = np.interp(i_coords, np.arange(len(y)), y)

plt.figure()
plt.plot(r_contour, y_contour, 'k.-')
plt.xlabel("r")
plt.ylabel("y")
plt.title(f"Surface level {level} of Re[ψ(r,y)]")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid(linestyle='--', alpha=0.5)
#%%
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, 'surfaces')
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
np.save("level16_200px.npy", Z)

