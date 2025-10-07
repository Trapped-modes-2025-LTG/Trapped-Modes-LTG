import numpy as np
import scipy.special as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tqdm import tqdm


class McIver1997:
    def __init__(self, a, c_index):        
        self.a = a
        self.c_values = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309]
        self.R = self.c_values[c_index] 

    def En(self, n, z):
        """Exponential integral generalizada."""
        if n == 1:
            return sp.exp1(z)
        else:
            return (np.exp(-z) - z * self.En(n - 1, z)) / (n - 1)

    def N(self, l, y, a):
        """Término analítico N(l, y, a) del paper."""
        return np.real(self.En(1, (l + 1j * y) * a)) + np.imag(self.En(2, (l + 1j * y) * a)) / a

    def integrando(self, nu, r, y, R):
        rplus = r * (r > R) + R * (r < R)
        rminus = r * (r < R) + R * (r > R)
        return (nu * np.cos(nu * y) - np.sin(nu * y)) * sp.iv(0, nu * rminus) * sp.kv(0, nu * rplus) * nu / (nu**2 + 1)

    def integral(self, r, y, R):
        return quad(self.integrando, 0, self.a, args=(r, y, R))[0]

    def phi(self, r, y):
        R = self.R
        a = self.a

        vectorized_integral = np.vectorize(self.integral)

        rplus = r * (r > R) + R * (r < R)
        rminus = r * (r < R) + R * (r > R)

        M = R * np.exp(-y) * sp.jv(0, rminus) * sp.hankel1(0, rplus)

        return 4j * (np.pi**2) * M + 8 * R * (vectorized_integral(r, y, R) + self.N(np.abs(r - R), y, a) / (2 * np.sqrt(r * R)))


if __name__ == "__main__":
    solver = McIver1997(a=30, c_index = 0)
    solver1 = McIver1997(a=30, c_index = 1)
    r = np.linspace(0.1, 10, 350)
    y = 0.0  
    
    phi_vals0 = np.array([solver.phi(ri, y) for ri in tqdm(r)], dtype=complex)
    phi_vals1 = np.array([solver1.phi(ri, y) for ri in tqdm(r)], dtype=complex)

    Z0 = np.real(phi_vals0)
    Z1 = np.real(phi_vals1)

    plt.figure()
    plt.plot(r,Z0, ".-", label = r"$c = j_{0,0}$")
    plt.plot(r,Z1, ".-", label = r"$c = j_{0,1}$")
    plt.xlabel("r")
    plt.legend()
    plt.vlines(x =  2.4048,ymin =min([np.min(Z0), np.min(Z1)])-5, ymax= max([np.max(Z0), np.max(Z1)])+5, linestyles = "--", color = "k", alpha = 0.5)
    plt.vlines(x =  5.5201,ymin =min([np.min(Z0), np.min(Z1)])-5, ymax= max([np.max(Z0), np.max(Z1)])+5, linestyles = "--", color = "k", alpha = 0.5)
    plt.ylim(ymin=min([np.min(Z0), np.min(Z1)])-5, ymax= max([np.max(Z0), np.max(Z1)])+5)
    plt.ylabel(r"$\phi$")
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("phi_c0_ci.pdf")
    plt.show()

