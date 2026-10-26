import sys
sys.path.append('cqToolbox')

import os
import math
import time
import numpy as np
import scipy.io
from linearcq import Conv_Operator
from graded_cq import apply_RKconvolfast_graded
from paper_style import paper_figure

## Figure 4 of the paper with the graded parsimonious RadauIIA-2 (ct = 0.01, branch 2 obtained
## from branch 1 by the symmetry s_2(l) = conj(s_1(1-l))). Problem as in the paper: g = sin^2,
## K(s) = s/(1-exp(-s)), T = 5, N = 64*2^j. Laplace evaluations are counted on one side of the
## real axis only, since the other half follows by conjugation: L = m*N for classic CQ.
## Reference lines: L^{-p} for classic CQ and L^{-p*2m/(2m-1)} for the parsimonious method, p = 3/4.
## Usage: python3 paperplot_RadauIIA2.py [--recompute]   (otherwise the saved data are plotted)
## The present Figure 4 shows the maximal error (although labelled as weighted l^2-error);
## ERROR_NORM selects which of the two saved errors is plotted.
ERROR_NORM = "max"
datafile = "../scalar/data/paper_RadauIIA2_graded.mat"
solfile  = "../scalar/data/paper_RadauIIA2_graded_sol.mat"

def create_timepoints(N,T):
	m = 2
	c_RK=np.array([1.0/3,1])
	time_points=np.zeros((1,m*N))
	for j in range(m):
		time_points[0,j:m*N:m]=c_RK[j]*1.0/N*np.ones((1,N))+np.linspace(0,1-1.0/N,N)
	return T*time_points

if "--recompute" in sys.argv or not os.path.exists(datafile):
	T=5
	g = lambda t : np.sin(t)**2
	def gd(t):
		I = t < 0
		res = 2*np.cos(t)*np.sin(t)
		res[I] = 0
		return res
	Ks = lambda s: s/(1-np.exp(-s))
	ScatOperator_classic=Conv_Operator(lambda s,b: Ks(s)*b)
	ScatOperator_graded=Conv_Operator(lambda s,b: Ks(s)*b)
	ct = 0.01
	Am = 10
	m = 2
	res = dict(N=np.zeros(Am),L_classic=np.zeros(Am),L_graded=np.zeros(Am),err_max=np.zeros(Am),errg_max=np.zeros(Am),
	           err_l2=np.zeros(Am),errg_l2=np.zeros(Am),dev_l2=np.zeros(Am),ct=ct)
	sols = dict()
	for j in range(Am):
		N = 64*2**j
		dt = T/N
		tt = np.linspace(0,T,N+1)
		ex_sol = gd(tt)+gd(tt-1)+gd(tt-2)+gd(tt-3)+gd(tt-4)
		rhs = g(create_timepoints(N,T))
		start = time.time()
		num_solStages = ScatOperator_classic.apply_RKconvol(rhs,T,cutoff=10**(-10),show_progress=False,method="RadauIIA-2",first_value_is_t0=False)
		num_solStagesgraded,count,info = apply_RKconvolfast_graded(ScatOperator_graded,rhs,T,Ks,method="RadauIIA-2",ct=ct,same_grading=True)
		solRK=np.zeros(N+1)
		solRKgraded=np.zeros(N+1)
		solRK[1:N+1]=np.real(num_solStages[0,m-1:N*m:m])
		solRKgraded[1:N+1]=np.real(num_solStagesgraded[0,m-1:N*m:m])
		res["N"][j] = N
		res["L_classic"][j] = m*N
		res["L_graded"][j] = count
		res["err_max"][j] = max(np.abs(solRK-ex_sol))
		res["errg_max"][j] = max(np.abs(solRKgraded-ex_sol))
		res["err_l2"][j] = np.sqrt(dt)*np.linalg.norm(solRK-ex_sol)
		res["errg_l2"][j] = np.sqrt(dt)*np.linalg.norm(solRKgraded-ex_sol)
		res["dev_l2"][j] = np.sqrt(dt)*np.linalg.norm(solRKgraded-solRK)
		sols["N"+str(N)] = dict(t=tt,ex_sol=ex_sol,solRK=solRK,solRKgraded=solRKgraded)
		print("N: ",N," L classic/graded: ",m*N,count," Errors max: ",res["err_max"][j],res["errg_max"][j],
		      " Deviation: ",res["dev_l2"][j]," Time: ",time.time()-start)
	scipy.io.savemat(datafile,res)
	scipy.io.savemat(solfile,sols)

d = scipy.io.loadmat(datafile)
v = lambda k: d[k].ravel()
paper_figure("RadauIIA-2",v("L_classic"),v("err_"+ERROR_NORM),v("L_graded"),v("errg_"+ERROR_NORM),
             3.0/4,r"$\mathcal{O}(L^{-3/4})$",v("N"),v("dev_l2"),"paper_errs_RadauIIA2.pdf",ref_shift=0.75,
             pars_ref=(3.0/4*4/3,r"$\mathcal{O}(L^{-1})$",0.6))
