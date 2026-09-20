import sys
sys.path.append('cqToolbox')

import numpy as np
import math
from linearcq import Conv_Operator

def create_timepoints(method,N,T):
	if (method=="RadauIIA-2"):
		m = 2
		c_RK=np.array([1.0/3,1])
	if (method=="RadauIIA-3"):
		m = 3
		c_RK=np.array([2.0/5-math.sqrt(6)/10,2.0/5+math.sqrt(6)/10,1])
	time_points=np.zeros((1,m*N))
	for j in range(m):
		time_points[0,j:m*N:m]=c_RK[j]*1.0/N*np.ones((1,N))+np.linspace(0,1-1.0/N,N)
	return T*time_points

## Creating right-hand side
T=5
## Frequency - domain operator defined
g = lambda t : np.sin(t)**2
def gd(t):
    I = t < 0
    res = 2*np.cos(t)*np.sin(t)
    res[I] = 0
    return res


# Hyperbolic kernel
#Ks = lambda s: s
Ks = lambda s: s/(1-np.exp(-s))
def freq_der(s,b):
	return Ks(s)*b
	#return s**(-1)*b

ScatOperator=Conv_Operator(freq_der)
ScatOperator_classic=Conv_Operator(freq_der)

Am = 7
taus = np.zeros(Am)
errRK = np.zeros(Am)
errRKfast = np.zeros(Am)
errBDF = np.zeros(Am)
#erraltBDF = np.zeros(Am)
for j in range(Am):
	N=64*2**j
	taus[j] = T/N
	tt=np.linspace(0,T,N+1)
	#ex_sol = 1.0/7*tt**7
	ex_sol = gd(tt)+gd(tt-1)+gd(tt-2)+gd(tt-3)+gd(tt-4)
	#ex_sol = gd(tt)
	#rhs = np.linspace(0,T,N+1)**6
	m = 2
	time_points=create_timepoints("RadauIIA-"+str(m),N,T)
	rhs=g(time_points)
	num_solStagesfast, freqs = ScatOperator.apply_RKconvolfast(rhs,T,Ks,cutoff=10**(-10),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False,return_freqs=True)
	num_solStages, freqs = ScatOperator_classic.apply_RKconvol(rhs,T,cutoff=10**(-10),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False,return_freqs=True)
	solRK=np.zeros(N+1)
	solRKfast=np.zeros(N+1)
	solRK[1:N+1]=np.real(num_solStages[0,m-1:N*m:m])
	solRKfast[1:N+1]=np.real(num_solStagesfast[0,m-1:N*m:m])

	errRK[j] = max(np.abs(solRK-ex_sol))
	errRKfast[j] = max(np.abs(solRKfast-ex_sol))

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(5)
plt.loglog(taus/m,taus**2,linestyle='dashed')
plt.loglog(taus/m,errRK,label="Classic RK")
plt.loglog(taus/m,errRKfast,label="Fast RK")
plt.legend()
#plt.loglog(taus,taus**3,linestyle='dashed')
plt.savefig('errs_RK.pdf')
plt.figure(12)
freqs = np.array(freqs)
for j in range(m):
	#plt.plot(np.real(freqs[:,j]))
	#plt.plot(np.imag(freqs[:,j]))
    plt.scatter(np.real(freqs[:,j]),np.imag(freqs[:,j]),label = "Stage "+str(j+1))
    #plt.scatter(np.real(freqs[:,j]),-np.imag(freqs[:,j]),label = "Stage "+str(j+1))
#plt.legend()
plt.savefig('freqs_RK.pdf')
plt.figure(13)
cut =-1
plt.plot(tt[:cut],solRK[:cut],label="Classic RK")
plt.plot(tt[:cut],solRKfast[:cut],label="Fast RK")
plt.plot(tt[:cut],ex_sol[:cut],linestyle='dashed',label="Exact solution")
plt.legend()
plt.savefig('sol_RK.pdf')