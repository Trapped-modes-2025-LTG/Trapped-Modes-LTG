import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad
from matplotlib.colors import Normalize


a = 100 # integral limit for aproximation
c =  2.4048 # zero of Bessel's funtion J_0 [2.4048, 5.5201, 8.6537, 11.7915, 14.9309]

def En(n, z):
    if n <= 0:
        raise Exception("n must be greater than 0.")
    elif n == 1:
        return sp.exp1(z)
    else:
        return (np.exp(-z) - z * En(n - 1, z)) / (n - 1)

def E1(z):
    return En(1, z)

def E2(z):
    return En(2, z)

def L(d, y, a):
    return -np.imag(E1((d + 1j * y) * a)) + np.real(E2((d + 1j * y) * a)) / a

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

#%%
'''
First for a psi that doesn´t evolve in time
'''

def psi(r, y):
    sol = np.zeros(np.shape(r), dtype=complex)
    for i in range(len(r[:, 0])):
        for j in range(len(r[0, :])):
            sol[i, j] = psi1(r[i, j], y[i, j]) if (r[i, j] < c) else psi2(r[i, j], y[i, j])
    return sol

r = np.linspace(0.01, 10, 100)
y = np.linspace(-1, 5, 100)
R, Y = np.meshgrid(r,y)
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
cs = plt.contour(R, Y, np.real(psi(R,Y)), levels = [20, 16, 12, 8, 4][::-1])

#%%
'''
Now for a psi that evolves in time with frequency w
'''

g = 9.81
a = 100
c = 2.4048
w = np.sqrt(g*c)


def psi(r, y):
    sol = np.zeros(np.shape(r), dtype=complex)
    for i in range(len(r[:, 0])):
        for j in range(len(r[0, :])):
            sol[i, j] = psi1(r[i, j], y[i, j]) if (r[i, j] < c) else psi2(r[i, j], y[i, j])
    return sol

# create a meshgrid as input for the funtion psi
r = np.linspace(0.01, 10, 100)
y = np.linspace(-1, 5, 100)
R, Y = np.meshgrid(r, y)

psi_base = psi(R, Y)

times = np.linspace(0, 3, 32)
contornos = []

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.invert_yaxis()

colors = plt.cm.viridis(np.linspace(0, 1, len(times)))  # color related to each time

for i, t in enumerate(times):
    psi_t = np.real(psi_base * np.exp(1j * w * t))
    cs = ax.contour(R, Y, psi_t, levels=[8], colors=[colors[i]])
    contornos.append(cs)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=times[0], vmax=times[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='Tiempo')
ax.set_title("Evolución temporal de la \n línea de corriente (nivel = 8)" )

plt.xlabel("r")
plt.xlim(0, 6)
plt.ylim(3, -1)
plt.ylabel("y")
plt.show()
