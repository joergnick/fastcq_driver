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

import scipy.io
import time

N=512*2**3
#N=512*2**9
dt = T/N
m = 2
taus = T/N
ScatOperator=Conv_Operator(freq_der,order=1)
freqs = ScatOperator.get_zeta_vect(N,T)

ScatOperator=Conv_Operator(freq_der,order=2)
freqs2 = ScatOperator.get_zeta_vect(N,T)
freqs3 = ScatOperator.gen_RK_frequencies(N,T,show_progress=False,method="RadauIIA-2")
freqs4 = ScatOperator.gen_RK_frequencies(N,T,show_progress=False,method="RadauIIA-3")
freqs3 = np.array(freqs3)[:,0]
freqs4 = np.array(freqs4)
Lhalf = len(freqs4[:,0])//2+1
freqs4 = freqs4[:,1]
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': True,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# MATLAB default color order
C1 = (0.0000, 0.4470, 0.7410)  # blue   — Implicit Euler
C2 = (0.8500, 0.3250, 0.0980)  # orange — BDF-2
C3 = (0.9290, 0.6940, 0.1250)  # yellow — RadauIIA-2
C4 = (0.4940, 0.1840, 0.5560)  # purple — RadauIIA-3

solRK=np.zeros(N+1)
solRKfast=np.zeros(N+1)

fig, ax = plt.subplots(num=12, figsize=(5.5, 4.0))

freqshlf = freqs[0:N//2]
freqshlf2 = freqs2[0:N//2]
freqshlf3 = freqs3[0:N//2]
freqshlf4 = freqs4[np.imag(freqs4)<0]

alpha = 0
lw = 1.8

ax.loglog(np.imag(freqshlf),  np.real(freqshlf)  - alpha, color=C1, linewidth=lw, label='Implicit Euler')
ax.loglog(np.imag(freqshlf2), np.real(freqshlf2) - alpha, color=C2, linewidth=lw, label='BDF-2')
ax.loglog(np.imag(freqshlf3), np.real(freqshlf3) - alpha, color=C3, linewidth=lw, label='RadauIIA-2')
ax.loglog(-np.imag(freqshlf4),np.real(freqshlf4) - alpha, color=C4, linewidth=lw, label='RadauIIA-3')

ax.loglog(np.imag(freqshlf),  5*1e-4 *np.imag(freqshlf) **2, color=C1, linewidth=1.0, linestyle='dashed', label=r'$\mathcal{O}(\mathrm{Im}\,s)^2$')
ax.loglog(np.imag(freqshlf2), 1e-7   *np.imag(freqshlf2)**3, color=C2, linewidth=1.0, linestyle='dashed', label=r'$\mathcal{O}(\mathrm{Im}\,s)^3$')
ax.loglog(np.imag(freqshlf3), 0.2*1e-7*np.imag(freqshlf3)**3,color=C3, linewidth=1.0, linestyle='dashed', label=r'$\mathcal{O}(\mathrm{Im}\,s)^3$')
ax.loglog(-np.imag(freqshlf4),0.8*1e-12*np.imag(freqshlf4)**4,color=C4,linewidth=1.0, linestyle='dashed', label=r'$\mathcal{O}(\mathrm{Im}\,s)^4$')

ax.set_xlim([1e1, 1e4])
ax.set_ylim([2, 1e3])
ax.set_xlabel(r'$\mathrm{Im}\,s$')
ax.set_ylabel(r'$\mathrm{Re}\,s$')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)

fig.tight_layout()
fig.savefig('freqs_.pdf', bbox_inches='tight', dpi=300)
