import sys
sys.path.append('cqToolbox')

import numpy as np
import math
from linearcq import Conv_Operator

## Creating right-hand side
T=5
# Hyperbolic kernel
#Ks = lambda s: s
Ks = lambda s: s/(1-np.exp(-s))
def freq_der(s,b):
	return Ks(s)*b
	#return s**(-1)*b

N=512*2**3

ScatOperator=Conv_Operator(freq_der,order=1)
L,dt,tol,lam = ScatOperator.get_integration_parameters(N,T)

def expr(phi_r, phi_i):
    return (1.5
            - 2 * lam * np.cos(phi_r) * np.exp(-phi_i)
            + 0.5 * lam**2 * np.cos(2 * phi_r) * np.exp(-2 * phi_i))
 
# Solve expr = 0 analytically as quadratic in u = exp(-phi_i):
#   0.5*lam^2*cos(2*phi_r)*u^2 - 2*lam*cos(phi_r)*u + 1.5 = 0
phi_r_arr = 0.9**np.linspace(0, 100, 4000)
a_arr = 0.5 * lam**2 * np.cos(2 * phi_r_arr)
b_arr = -2 * lam * np.cos(phi_r_arr)
disc_arr = b_arr**2 - 4 * a_arr * 1.5
sqrt_disc = np.sqrt(np.maximum(disc_arr, 0.0))

with np.errstate(divide='ignore', invalid='ignore'):
    u1 = (-b_arr + sqrt_disc) / (2 * a_arr)
    u2 = (-b_arr - sqrt_disc) / (2 * a_arr)

phi_i1 = np.where((u1 > 0) & np.isfinite(u1), -np.log(u1), np.nan)
phi_i2 = np.where((u2 > 0) & np.isfinite(u2), -np.log(u2), np.nan)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(phi_r_arr, phi_r_arr**2, s=1, color='C0')
plt.scatter(phi_r_arr, phi_r_arr**3, s=1, color='C0')
plt.scatter(phi_r_arr, phi_r_arr**4, s=1, color='C0')

all_phi_i = np.concatenate([phi_i1[np.isfinite(phi_i1)], phi_i2[np.isfinite(phi_i2)]])
global_shift = np.min(-all_phi_i)
for phi_i_branch in [phi_i1, phi_i2]:
    mask = np.isfinite(phi_i_branch)
    plt.scatter(phi_r_arr[mask], 7 * ((-phi_i_branch[mask]) - global_shift), s=1, color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-3,0.5])
plt.ylim([1e-10,1])
plt.xlabel(r'$\varphi_r$')
plt.ylabel(r'$\varphi_i$')
plt.title(rf'Boundary  ($\lambda={lam}$)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('domain_analyticity.pdf', bbox_inches='tight', dpi=300)
