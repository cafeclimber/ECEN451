import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad
from numpy import linalg

N = 101

# Initialize parameters
freq       = 300e+6      # 300 MHz
wavelength = 3e+8 / freq # c / f
length     = 0.47 * wavelength
radius     = 0.005 * wavelength
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
print("Input Imedance: %s" % Z_in)

I_0 = I[feedpoint]

####################################################
#                                                  #
#           New Code for Antenna Patterns          #
#                                                  #
####################################################

thetas = np.linspace(-np.pi, np.pi, 500) # Make 500 evenly spaced thetas 

r = 2 # Far Field
def e_field(thetas):
    constant = 1j * eta * (k_0 * np.exp(-1j*k_0*r)) / (4*np.pi*r)
    E_theta = np.zeros((len(thetas),1), complex)
    for ind in range(len(thetas)):
        for z in range(len(pos)):
            integrand = I[z] * np.exp(1j*k_0*pos[z]*np.cos(thetas[ind]))
            E_theta[ind] += constant*np.sin(thetas[ind])*integrand

    E_theta = E_theta
    return E_theta

E_theta = e_field(thetas)
W_av = 1 / (2*eta) * (E_theta**2)
U = np.real((r**2) * W_av)

U_norm = U / np.amax(U)

U_max = np.amax(U_norm) * np.ones(len(thetas))

fig = plt.figure(figsize=(12,8.5))

xy = fig.add_subplot(222, projection='polar')
xz = fig.add_subplot(223, projection='polar')
yz = fig.add_subplot(224, projection='polar')
current = fig.add_subplot(221)

# Fonts
font = {'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

# Make plots nice and pretty
current.plot(pos, np.real(I))
xy.plot(thetas, np.real(U_max), linewidth='2.5')
xz.plot(thetas, np.real(U_norm), linewidth='2.5')
yz.plot(thetas, np.real(U_norm), linewidth='2.5')

# Set ticks
thetaticks = np.arange(0, 360, 30)
xy.set_thetagrids(thetaticks, frac=1.05)
xz.set_thetagrids(thetaticks, frac=1.05)
yz.set_thetagrids(thetaticks, frac=1.05)
yticks = np.arange(0, 1, 0.25)
xy.set_yticklabels(yticks)
xz.set_yticklabels(yticks)
yz.set_yticklabels(yticks)

# Add polarizations
xy.text(2.8, 1.25, 'Phi polarized')
xz.text(2.8, 1.25, 'Theta polarized')
yz.text(2.8, 1.25, 'Theta polarized')

current.set_title("Current Distribution")
xy.set_title("XY Cut Plane")
xz.set_title("XZ Cut Plane")
yz.set_title("YZ Cut Plane")

current.grid(True)
current.axhline(0, color='black')
current.axvline(0, color='black')

# Rotate so 0 degrees is at top
xy.set_theta_zero_location("N")
xy.set_theta_direction(-1)
xz.set_theta_zero_location("N")
xz.set_theta_direction(-1)
yz.set_theta_zero_location("N")
yz.set_theta_direction(-1)

fig.tight_layout()
plt.show()
