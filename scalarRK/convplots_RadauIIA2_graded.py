import sys
sys.path.append('cqToolbox')

import numpy as np
import math
import time
import scipy.io
import matplotlib.pyplot as plt
from linearcq import Conv_Operator
from rkmethods import RKMethod
from tcheby_helpers import tchebycheff_nodes,tcheb_interpolation

from graded_cq import graded_mesh,apply_RKconvolfast_graded

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

## Same problem as in convplots_RadauIIA2.py
T=5
varepsilon = 10**(-1)
g = lambda t : np.sin(t/varepsilon)**2
def gd(t):
    I = t < 0
    res = 2/varepsilon*np.cos(t/varepsilon)*np.sin(t/varepsilon)
    res[I] = 0
    return res

Ks = lambda s: s/(1-np.exp(-s))
def freq_der(s,b):
	return Ks(s)*b

ScatOperator=Conv_Operator(freq_der)
ScatOperator_classic=Conv_Operator(freq_der)
ScatOperator_graded=Conv_Operator(freq_der)

## Variants of the graded approach: (ct, same_grading)
variants = [(ct,same) for ct in [0.01,0.1,1.0] for same in [False,True]]
def variant_label(ct,same):
	return r"$\widetilde{c} = $"+str(ct)+(", same grading (conj. symmetry)" if same else ", mirrored")

Am = 12
m = 2
taus = np.zeros(Am)
errRK = np.zeros(Am)
errRKfast = np.zeros(Am)
devRKfast = np.zeros(Am)
count_evals = np.zeros(Am)
count_classic_evals = np.zeros(Am)
errRKgraded = np.zeros((len(variants),Am))
devRKgraded = np.zeros((len(variants),Am))
count_graded_evals = np.zeros((len(variants),Am))
infos = dict()
for j in range(Am):
	N=64*2**j
	## Laplace evaluations are counted only on one side of the real axis, since every scheme can
	## obtain the other side by conjugation; hence the counts of all schemes except the one
	## with same grading (which already exploits this) are halved.
	count_classic_evals[j] = N*m
	taus[j] = T/N
	tt=np.linspace(0,T,N+1)
	ex_sol = gd(tt)+gd(tt-1)+gd(tt-2)+gd(tt-3)+gd(tt-4)
	time_points=create_timepoints("RadauIIA-"+str(m),N,T)
	rhs=g(time_points)
	start = time.time()
	num_solStagesfast, freqs, count_evals[j] = ScatOperator.apply_RKconvolfast(rhs,T,Ks,cutoff=10**(-10),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False,return_freqs=True)
	count_evals[j] = count_evals[j]/2
	num_solStages =ScatOperator_classic.apply_RKconvol(rhs,T,cutoff=10**(-10),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False)
	solRK=np.zeros(N+1)
	solRKfast=np.zeros(N+1)
	solRK[1:N+1]=np.real(num_solStages[0,m-1:N*m:m])
	solRKfast[1:N+1]=np.real(num_solStagesfast[0,m-1:N*m:m])
	errRK[j] = max(np.abs(solRK-ex_sol))
	errRKfast[j] = max(np.abs(solRKfast-ex_sol))
	devRKfast[j] = max(np.abs(solRKfast-solRK))
	print("N: ", N, " L classic/fast: ", count_classic_evals[j], count_evals[j], " Errors classic/fast: ", errRK[j], errRKfast[j])
	for v,(ct,same) in enumerate(variants):
		num_solStagesgraded, count_graded_evals[v,j], info = apply_RKconvolfast_graded(ScatOperator_graded,rhs,T,Ks,method="RadauIIA-"+str(m),ct=ct,same_grading=same)
		if not same:
			count_graded_evals[v,j] = count_graded_evals[v,j]/2
		infos[(ct,same,N)] = info
		solRKgraded=np.zeros(N+1)
		solRKgraded[1:N+1]=np.real(num_solStagesgraded[0,m-1:N*m:m])
		errRKgraded[v,j] = max(np.abs(solRKgraded-ex_sol))
		devRKgraded[v,j] = max(np.abs(solRKgraded-solRK))
		print("    ct: ", ct, " same grading: ", same, " L: ", count_graded_evals[v,j], " Error: ", errRKgraded[v,j],
		      " Deviation from classic: ", devRKgraded[v,j], " Graded pieces: ", len(info["Phi"])-1)
	end = time.time()
	print("    Time: ", end-start)

res = dict()
res["Ns"]       = count_classic_evals
res["counters"] = count_evals
res["counters_graded"] = count_graded_evals
res["variants"] = np.array([[ct,same] for (ct,same) in variants])
res["taus"]     = taus
res["err"]      = errRK
res["errf"]     = errRKfast
res["errg"]     = errRKgraded
res["devf"]     = devRKfast
res["devg"]     = devRKgraded
scipy.io.savemat("../scalar/data/err_RadauIIA2_graded_HS.mat", res)

plt.close('all')
plt.rc('text', usetex=True)
plt.figure(figsize=(16, 5.5))
colors = {0.01:'C2',0.1:'C3',1.0:'C4'}
plt.subplot(1,3,1)
plt.loglog(count_classic_evals,errRK,label="Radau IIA-2",marker='s',color='C0')
plt.loglog(count_evals,errRKfast,label="Modified Radau IIA-2",marker='o',color='C1')
for v,(ct,same) in enumerate(variants):
	plt.loglog(count_graded_evals[v],errRKgraded[v],label=variant_label(ct,same),marker='d',markersize=4,color=colors[ct],linestyle='dashed' if same else 'solid')
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel("Maximal error")
plt.title("Error against exact solution")
plt.grid(True)
plt.legend(fontsize=8)
plt.subplot(1,3,2)
plt.loglog(count_evals,devRKfast,label="Modified Radau IIA-2",marker='o',color='C1')
for v,(ct,same) in enumerate(variants):
	plt.loglog(count_graded_evals[v],devRKgraded[v],label=variant_label(ct,same),marker='d',markersize=4,color=colors[ct],linestyle='dashed' if same else 'solid')
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel(r"$\max |u_{\mathrm{cq}}-u_{\mathrm{fast}}|$")
plt.title("Deviation from classic Radau IIA-2")
plt.grid(True)
plt.subplot(1,3,3)
Nmax = 64*2**(Am-1)
info_m = infos[(0.1,False,Nmax)]
info_s = infos[(0.1,True,Nmax)]
I = info_m["lls"] <= 0.5
plt.semilogy(np.pi*info_m["lls"][I],info_m["Kerr"][0,I],label="Branch 1",color='C0')
plt.semilogy(np.pi*info_m["lls"][I],info_m["Kerr"][1,I],label="Branch 2, mirrored",color='C1')
plt.semilogy(np.pi*info_s["lls"][I],info_s["Kerr"][1,I],label="Branch 2, same grading (conj. symmetry)",color='C2',alpha=0.7)
plt.axvline(np.pi/5,color='gray',linestyle='dashed')
plt.xlabel(r"$\Phi = \pi\, l/L$")
plt.ylabel("Interpolation error of $K$")
plt.title(r"$\widetilde{c} = 0.1$, $N = $"+str(Nmax))
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('errs_RK_RadauIIA2_graded.pdf')
