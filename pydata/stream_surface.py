import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad
from matplotlib.colors import Normalize

class PsiSolver:
    def __init__(self, a, c_index):        
        self.a = a
        self.c_values = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309]
        self.selected_c = self.c_values[c_index]  

    def En(self, n, z):
        if n == 1:
            return sp.exp1(z)
        else:
            return (np.exp(-z) - z * self.En(n - 1, z)) / (n - 1)

    def L(self, d, y):
        return -np.imag(self.En(1, (d + 1j * y) * self.a)) + np.real(self.En(1, (d + 1j * y) * self.a)) / self.a
    
    def integrand(self, nu, r, y, n):
        if n == 1:
            return (nu * np.sin(nu * y) + np.cos(nu * y)) * sp.iv(1, nu * r) * sp.kv(0, nu * self.selected_c) * nu / (nu ** 2 + 1)
        else:
            return (nu * np.sin(nu * y) + np.cos(nu * y)) * sp.iv(0, nu * self.selected_c) * sp.kv(1, nu * r) * nu / (nu ** 2 + 1)

    def integral(self, n, r, y):
        return quad(self.integrand, 0, self.a, args=(r, y, n))[0]
    
    def psi_n(self, n, r, y):
        if n == 1:
            M = lambda r, y: r * np.exp(-y) * sp.jv(1, r) * sp.hankel1(0, self.selected_c)
            return -4 * np.pi ** 2 * 1j * self.selected_c * M(r, y) - 8 * self.selected_c * r * self.integral(1, r, y) + \
                   self.L(np.abs(r - self.selected_c), y) / (2 * np.sqrt(r * self.selected_c))
        else:
            M = lambda r, y: r * np.exp(-y) * sp.jv(0, self.selected_c) * sp.hankel1(1, r)
            return -4 * np.pi ** 2 * 1j * self.selected_c * M(r, y) + 8 * self.selected_c * r * self.integral(2, r, y) + \
                   self.L(np.abs(r - self.selected_c), y) / (2 * np.sqrt(r * self.selected_c))

    def psi(self, r, y):
        sol = np.zeros(np.shape(r), dtype=complex)
        for i in range(len(r[:, 0])):
            for j in range(len(r[0, :])):
                sol[i, j] = self.psi_n(1, r[i, j], y[i, j]) if (r[i, j] < self.selected_c) else self.psi_n(2, r[i, j], y[i, j])
        return sol
    
def freq_mode(r,c_index,H,sigma = 0.0728 ,rho = 1002,eq = 0, g = 9.81):
    '''
    El cero de la función de Bessel J(c) = 0 corresponde al radio del anillo.
    Como el problema está adimensionalizado,  kr = c --> k = c/r.
    r es el parámetro con el que armamos los toroides en 3D, corresponde a aproximadamente el radio interno.
    Devuelve los omegas, no las frecuencias temporales.
    '''
    c = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309]
    k = c[c_index]/r
    def gc(k):
        return np.sqrt(k*(g+sigma*k**2/rho)*np.tanh(H*k))
    def only_g(k):
        return np.sqrt(g*k)
    
    return only_g(k) if eq==1 else gc(k)
    
if __name__ == "__main__":
    # First for a psi that doesn´t evolve in time

    solver = PsiSolver(a=100, c_index=0)
    r = np.linspace(0.01, 10, 100)
    y = np.linspace(-1, 5, 100)
    R, Y = np.meshgrid(r, y)
    
    psi = solver.psi(R, Y)
    Z = np.real(psi)
    
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    cs = plt.contour(R, Y, Z, levels = [20, 16, 12, 8, 4][::-1])
    plt.grid(linestyle = '--', alpha = 0.5)
    
    # Now for a psi that evolves in time with trapped mode's frequency w 
    
    g = 9.81
    w = np.sqrt(g*solver.selected_c)
    
    times = np.linspace(0, 3, 32)
    contornos = []
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))  # color related to each time
    
    for i, t in enumerate(times):
        psi_t = np.real(psi * np.exp(1j * w * t))
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
    plt.grid(linestyle = '--', alpha = 0.5)
    plt.show()
