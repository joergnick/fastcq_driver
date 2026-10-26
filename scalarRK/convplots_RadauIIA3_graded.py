import sys
sys.path.append('cqToolbox')

import numpy as np
import math
import time
import scipy.io
import matplotlib.pyplot as plt
from linearcq import Conv_Operator
from rkmethods import RKMethod
from contour_helpers import graded_mesh,interpolate_on_pieces

from graded_cq import apply_RKconvolfast_graded3

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

## Same problem as in convplots_RadauIIA3.py
T=5
g = lambda t : np.sin(t)**2
def gd(t):
    I = t < 0
    res = 2*np.cos(t)*np.sin(t)
    res[I] = 0
    return res

Ks = lambda s: s/(1-np.exp(-s))
def freq_der(s,b):
	return Ks(s)*b

ScatOperator=Conv_Operator(freq_der)
ScatOperator_classic=Conv_Operator(freq_der)
ScatOperator_graded=Conv_Operator(freq_der)

## Usage: python3 convplots_RadauIIA3_graded.py [ct], default ct = 0.01
ct = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
suffix = "" if ct == 0.01 else "_ct"+str(ct)
Am = 12
m = 3
taus = np.zeros(Am)
errRK = np.zeros(Am)
errRKfast = np.zeros(Am)
errRKgraded = np.zeros(Am)
devRKfast = np.zeros(Am)
devRKgraded = np.zeros(Am)
count_evals = np.zeros(Am)
count_graded_evals = np.zeros(Am)
count_classic_evals = np.zeros(Am)
sols = dict()
infos = []
for j in range(Am):
	N=16*2**j
	dt = T/N
	count_classic_evals[j] = N*m
	taus[j] = T/N
	tt=np.linspace(0,T,N+1)
	ex_sol = gd(tt)+gd(tt-1)+gd(tt-2)+gd(tt-3)+gd(tt-4)
	time_points=create_timepoints("RadauIIA-"+str(m),N,T)
	rhs=g(time_points)
	start = time.time()
	ScatOperator.tol = max(1e-14, 0.01*taus[j]**4)
	ScatOperator_classic.tol = ScatOperator.tol
	ScatOperator_graded.tol = ScatOperator.tol
	num_solStagesfast, freqs, count_evals[j] = ScatOperator.apply_RKconvolfast(rhs,T,Ks,cutoff=10**(-10),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False,return_freqs=True)
	count_evals[j] = count_evals[j]/2
	num_solStages = ScatOperator_classic.apply_RKconvol(rhs,T,cutoff=10**(-10),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False)
	num_solStagesgraded, count_graded_evals[j], info = apply_RKconvolfast_graded3(ScatOperator_graded,rhs,T,Ks,ct=ct)
	infos.append(info)
	solRK=np.zeros(N+1)
	solRKfast=np.zeros(N+1)
	solRKgraded=np.zeros(N+1)
	solRK[1:N+1]=np.real(num_solStages[0,m-1:N*m:m])
	solRKfast[1:N+1]=np.real(num_solStagesfast[0,m-1:N*m:m])
	solRKgraded[1:N+1]=np.real(num_solStagesgraded[0,m-1:N*m:m])
	errRK[j] = np.sqrt(dt)*np.linalg.norm(np.abs(solRK-ex_sol))
	errRKfast[j] = np.sqrt(dt)*np.linalg.norm(np.abs(solRKfast-ex_sol))
	errRKgraded[j] = np.sqrt(dt)*np.linalg.norm(np.abs(solRKgraded-ex_sol))
	devRKfast[j] = np.sqrt(dt)*np.linalg.norm(np.abs(solRKfast-solRK))
	devRKgraded[j] = np.sqrt(dt)*np.linalg.norm(np.abs(solRKgraded-solRK))
	sols["N"+str(N)] = dict(t=tt,ex_sol=ex_sol,solRK=solRK,solRKfast=solRKfast,solRKgraded=solRKgraded)
	end = time.time()
	print("N: ", N, " Time: ", end-start, " L classic/fast/graded: ", count_classic_evals[j], count_evals[j], count_graded_evals[j],
	      " Errors: ", errRK[j], errRKfast[j], errRKgraded[j], " Deviation from classic fast/graded: ", devRKfast[j], devRKgraded[j],
	      " Graded pieces: ", len(info["Phi"])-1)

res = dict()
res["Ns"]       = count_classic_evals
res["counters"] = count_evals
res["counters_graded"] = count_graded_evals
res["taus"]     = taus
res["err"]      = errRK
res["errf"]     = errRKfast
res["errg"]     = errRKgraded
res["devf"]     = devRKfast
res["devg"]     = devRKgraded
res["ct"]       = ct
scipy.io.savemat("../scalar/data/err_RadauIIA3_graded_HS"+suffix+".mat", res)
scipy.io.savemat("../scalar/data/sol_RadauIIA3_graded_HS"+suffix+".mat", sols)

plt.close('all')
plt.rc('text', usetex=True)
plt.figure(figsize=(16, 5.5))
plt.subplot(1,3,1)
plt.loglog(count_classic_evals,errRK,label="Radau IIA-3",marker='s',color='C0')
plt.loglog(count_evals,errRKfast,label="Modified Radau IIA-3",marker='o',color='C1')
plt.loglog(count_graded_evals,errRKgraded,label=r"Modified Radau IIA-3, graded, $\widetilde c = %s$"%ct,marker='d',color='C2')
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel("Error")
plt.title("Error against exact solution")
plt.grid(True)
plt.legend(fontsize=8)
plt.subplot(1,3,2)
plt.loglog(count_evals,devRKfast,label="Modified Radau IIA-3",marker='o',color='C1')
plt.loglog(count_graded_evals,devRKgraded,label=r"Graded, $\widetilde c = %s$"%ct,marker='d',color='C2')
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel(r"$\|u_{\mathrm{cq}}-u_{\mathrm{fast}}\|$")
plt.title("Deviation from classic Radau IIA-3")
plt.grid(True)
plt.legend(fontsize=8)
plt.subplot(1,3,3)
info = infos[-1]
I = info["lls"] <= 0.5
for b in range(m):
	plt.semilogy(np.pi*info["lls"][I],info["Kerr"][b,I],label="Branch "+str(b))
plt.axvline(0.34*np.pi,color='gray',linestyle='dashed')
plt.xlabel(r"$\Phi = \pi\, l/L$")
plt.ylabel("Interpolation error of $K$")
plt.title(r"$\widetilde c = %s$, $N = %d$"%(ct,16*2**(Am-1)))
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('errs_RK_RadauIIA3_graded'+suffix+'.pdf')
