import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad
from matplotlib.colors import Normalize
from tqdm import tqdm

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
    
    def integrand_phi(self, nu, r, y):
        i = (nu * np.sin(nu * y) + np.cos(nu * y)) * sp.iv(0, nu * r) * sp.kv(0, nu * self.selected_c) * nu / (nu ** 2 + 1)
        return i 

    def integral(self, n, r, y):
        return quad(self.integrand, 0, self.a, args=(r, y, n))[0]
    
    def integral_phi(self, r, y):
        return quad(self.integrand_phi, 0, self.a, args=(r, y))[0]
    
    def psi_n(self, n, r, y):
        if n == 1:
            M = lambda r, y: r * np.exp(-y) * sp.jv(1, r) * sp.hankel1(0, self.selected_c)
            return -4 * np.pi ** 2 * 1j * self.selected_c * M(r, y) - 8 * self.selected_c * r * self.integral(1, r, y) + \
                   self.L(np.abs(r - self.selected_c), y) / (2 * np.sqrt(r * self.selected_c))
        else:
            M = lambda r, y: r * np.exp(-y) * sp.jv(0, self.selected_c) * sp.hankel1(1, r)
            return -4 * np.pi ** 2 * 1j * self.selected_c * M(r, y) + 8 * self.selected_c * r * self.integral(2, r, y) + \
                   self.L(np.abs(r - self.selected_c), y) / (2 * np.sqrt(r * self.selected_c))
                      
    def phi_n(self,r, y):
        phi_n = 8*self.selected_c*self.integral_phi(r, y) + \
            self.L(np.abs(r - self.selected_c), y) / (2 * np.sqrt(r * self.selected_c))
        return phi_n
    
    def psi_s(self, r, y):
        sol = np.zeros(np.shape(r), dtype=complex)
        for i in tqdm(range(len(r[:, 0]))):
            for j in range(len(r[0, :])):
                sol[i, j] = self.psi_n(1, r[i, j], y[i, j]) if (r[i, j] < self.selected_c) else self.psi_n(2, r[i, j], y[i, j])
        return sol
    
    def phi(self,r,y):
        sol = np.zeros(np.shape(r), dtype=complex)
        for i in tqdm(range(len(r[:, 0]))):
            for j in range(len(r[0, :])):
                sol[i, j] = self.phi_n( r[i, j], y[i, j])
        return sol
    
    
    def P_l(self,cosh_a,nu):
        return sp.eval_legendre(nu,cosh_a)
    
    def cosh_tau(self, r, y):   # alpha/tau >= 0
        
        d1 = np.sqrt((r + self.selected_c)**2 + y**2 ) 
        d2 = np.sqrt((r - self.selected_c)**2 + y**2 ) 
        num = d1**2 + d2**2
        den = 2*d1*d2
        val = num/den 

        return val 
    
    def sigma(self, r, y): # -pi/0 < sigma/beta <= pi
        
        num = r**2-self.selected_c**2+y**2
        den = np.sqrt((r**2-self.selected_c**2+y**2)**2 + 4*(self.selected_c**2)*(y**2))
        
        # return np.sign(y)*np.arccos(num/den)
        return np.arccos(num/den)
    
    def psi_d(self, r, y):
        cosh_a = self.cosh_tau(r, y)
        b = self.sigma(r, y)
        c = self.selected_c
        
        P12  = self.P_l(cosh_a,  1/2)
        Pm12 = self.P_l(cosh_a, -1/2)
        P32  = self.P_l(cosh_a,  3/2)
    
        cos_b = np.cos(b)
        sin_b  = np.sin(b)
    
        H = (1-cosh_a*cos_b)/(cosh_a-cos_b)
    
        psi = c * np.sqrt(cosh_a-cos_b)*(c*(H*P12-Pm12)+(3/2)*sin_b*(H*P32-Pm12))
        
        return psi
        
    def psi(self, r, y, delta):
        sigma = np.sqrt(2)*np.pi/(self.selected_c)**2
        psi = -1/2*r**2 - delta*(self.psi_s(r, y)+ sigma * self.psi_d(r, y))
            
        return psi
    
def freq_mode(r,c_index,H,sigma = 0.0728 ,rho = 1002,eq = "gravedad", g = 9.81):
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
    
    if eq == "gravedad":
        return only_g(k)
    elif eq == "capilar":
        return gc(k)
    else: 
        raise ValueError("eq must me gravedad or capilar")
    
if __name__ == "__main__":

    solver = PsiSolver(a=100, c_index=1)
    r = np.linspace(0.01, 9,100)
    y = np.linspace(-0.1, 4, 100)
    R, Y = np.meshgrid(r, y)
    
    # Cross section \psi
    
    psi = solver.psi(R, Y, delta = 5)
    Z = np.real(psi)
    
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    
    levels = list(np.linspace(start= -20,stop = -40, num = 20))
    cs = plt.contour(R, Y, Z, levels = levels[::-1], linestyles = "-")
    
    plt.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
    
    plt.grid(linestyle = '--', alpha = 0.5)
    plt.vlines(x = 5.52, ymin = -0.1, ymax = 4, alpha = 0.5, color = "k", linestyle = "-")
    plt.tight_layout()
    plt.ylabel("z (u.a.)")
    plt.xlabel("r (u.a.)")
    plt.show() 
    # Potential \psi
    
    # phi = solver.phi(R, Y)
    
##%%
#
#plt.figure()
#plt.gca().set_aspect('equal')
#plt.gca().invert_yaxis()
#
#levels = list(np.linspace(start= -20,stop = -40, num = 20))
#cs = plt.contour(R, Y, Z, levels = levels[::-1], linestyles = "-")
#plt.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
## delta = 5
#levels1 = list(np.linspace(start= -33.2799,stop = -33.28, num = 1))
#cs1 = plt.contour(R, Y, Z, levels = levels1[::-1], linestyles = "-", colors = "k", linewidths=3)
#
#
##%%
## Extract the points of the contour
#path = cs1.collections[0].get_paths()[0]
#vertices = path.vertices
#x, z = vertices[:, 0], vertices[:, 1]
#
#import ezdxf
#
## Cargar los puntos del CSV (si ya lo tenés guardado)
## Si ya tenés x y z en memoria, podés omitir esta línea
#x, z = np.loadtxt("contour_line_3cm.csv", delimiter=",", skiprows=1, unpack=True)
#
## Crear nuevo archivo DXF
#doc = ezdxf.new(dxfversion='R2010')
#msp = doc.modelspace()
#
## Agregar spline con los puntos (x,z)
#points = list(zip(x, z))
#msp.add_spline(points)
#
## Guardar como DXF
#doc.saveas("contour_line_3cm.dxf")
#print("✅ Archivo guardado como contour_line_3cm.dxf")
