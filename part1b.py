import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad
from numpy import linalg

N = int(input("Number of segments:"))

if N % 2 == 0:
    print("N must be odd")
    print("Changing %d to %d" % (N, N+1))
    N = N+1

# Initialize parameters
freq       = 300e+6      # 300 MHz
wavelength = 3e+8 / freq # c / f
length     = 0.47 * wavelength
radius     = (0.005/2) * wavelength
mu_0       = (4*np.pi)*(10**-7)
ep_0       = 8.8542e-12

k_0        = (2*np.pi) / wavelength
eta        = np.sqrt(mu_0/ep_0)
delta_z    = length / N
feedpoint  = int(np.floor(N/2))

z_scale = 1j * ((delta_z*eta) / k_0)

# Initialize vectors and matrices
V   = np.zeros((N,1), complex)
I   = np.zeros((N,1), complex)
Z   = np.zeros((N,N), complex)
pos = np.zeros((N,1))

# Calculate positions to perform integration
for m in range(N):
    pos[m] = (0.5 * delta_z) + ((m-1)*delta_z) - (length / 2)

# Define helper functions
def R(z_m, z_prime):
    return np.sqrt( ((z_m - z_prime)**2) + radius**2)

def r_integrand(z_prime, z_m):
    R_eval = R(z_m, z_prime)
    return sp.real((np.exp( -1j*k_0*R_eval)) / (R_eval))

def i_integrand(z_prime, z_m):
    R_eval = R(z_m, z_prime)
    return sp.imag((np.exp( -1j*k_0*R_eval)) / (R_eval))

def calc_integral(z_m, z_n):
    low  = z_n - (delta_z/2)
    high = z_n + (delta_z/2)
    
    real_part = quad(r_integrand, low, high, args=(z_m))
    imag_part = quad(i_integrand, low, high, args=(z_m))

    return real_part[0] + 1j*imag_part[0]

def calc_add_term(z_m, z_n):
    def calc_partial(z_m, z_prime):
        R_eval = R(z_m, z_prime)
        term1 = (z_m - z_prime)
        term2 = (1+1j*k_0*R_eval) / (R_eval**3)
        term3 = np.exp(-1j*k_0*R_eval)

        return term1 * term2 * term3

    z_prime_low  = z_n - (delta_z/2)
    z_prime_high = z_n + (delta_z/2)

    high = calc_partial(z_m, z_prime_high)
    low  = calc_partial(z_m, z_prime_low)

    return high - low

def impedance(m, n):
    z_m = pos[m]
    z_n = pos[n]

    integral = calc_integral(z_m, z_n)
    addition = calc_add_term(z_m, z_n)

    return ((k_0**2 / (4*np.pi)) * integral) + (addition[0] / (4*np.pi))

# Generate impedance matrix
for m in range(N):
    for n in range(N):
        Z[m][n] = impedance(m, n)

Z = z_scale * Z
Y = linalg.inv(Z)
V[feedpoint] = 1
I = np.dot(Y,V)
Z_in = (V[feedpoint] / I[feedpoint])
print(Z_in)

# Make the plot pretty
plt.xlim(pos.min()*1.1, pos.max()*1.1)
plt.title("Current Distribution on a Dipole")
plt.xlabel("Position [wavelengths]", fontsize = 16)
plt.ylabel("Current [A]", fontsize = 16)
plt.grid(True)

plt.plot(pos, np.absolute(I))
plt.show()
