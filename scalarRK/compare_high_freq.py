import sys
sys.path.append('cqToolbox')
sys.path.append('../scalar')

import numpy as np
import math
import time
import scipy.io
import matplotlib.pyplot as plt

from linearcq import Conv_Operator
from cq_helpers import evalCQ, evalfastCQ, BDF1, BDF2, TR

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
T=3
## Frequency - domain operator defined
varepsilon = 10**(-2)
g = lambda t : np.sin(t/varepsilon)**2
def gd(t):
    I = t < 0
    res = 2/varepsilon*np.cos(t/varepsilon)*np.sin(t/varepsilon)
    res[I] = 0
    return res


# Hyperbolic kernel
#Ks = lambda s: s
Ks = lambda s: s/(1-np.exp(-s))
def freq_der(s,b):
	return Ks(s)*b
	#return s**(-1)*b

Am = 10
#Am = 10
# Implicit Euler only converges at order 1, so its convergence needs many
# more timesteps than the higher order methods to become visible on the
# same log-log plot.
Am_ie = 13
taus = np.zeros(Am)
res = dict()

# ---------------------------------------------------------------------------
# Runge-Kutta collocation methods (RadauIIA-2, RadauIIA-3): classic vs. fast
# (parsimonious/interpolated) convolution quadrature, using the RK machinery
# in cqToolbox/linearcq.py.
# ---------------------------------------------------------------------------
rk_methods = ["RadauIIA-2","RadauIIA-3"]
rk_results = {}
for method in rk_methods:
	m = 2 if method=="RadauIIA-2" else 3
	ScatOperator=Conv_Operator(freq_der)
	ScatOperator_classic=Conv_Operator(freq_der)
	errRK = np.zeros(Am)
	errRKfast = np.zeros(Am)
	errsCQ_vs_fastCQ = np.zeros(Am)
	count_evals = np.zeros(Am)
	count_classic_evals = np.zeros(Am)
	for j in range(Am):
		N=256*2**j
		dt = T/N
		count_classic_evals[j] = 2*N*m
		taus[j] = T/N
		tt=np.linspace(0,T,N+1)
		ex_sol = gd(tt)+gd(tt-1)+gd(tt-2)+gd(tt-3)+gd(tt-4)
		time_points=create_timepoints(method,N,T)
		rhs=g(time_points)
		if m == 3:
			# RadauIIA-3 needs a looser tolerance at high N to stay well
			# conditioned, see convplots_RadauIIA3.py.
			ScatOperator.tol = max(1e-14, 0.01*dt**4)
			ScatOperator_classic.tol = ScatOperator.tol
		start = time.time()
		num_solStagesfast, freqs, count_evals[j] = ScatOperator.apply_RKconvolfast(rhs,T,Ks,cutoff=10**(-10),show_progress=False,method=method,first_value_is_t0=False,return_freqs=True)
		num_solStages, freqs = ScatOperator_classic.apply_RKconvol(rhs,T,cutoff=10**(-10),show_progress=False,method=method,first_value_is_t0=False,return_freqs=True)
		solRK=np.zeros(N+1)
		solRKfast=np.zeros(N+1)
		solRK[1:N+1]=np.real(num_solStages[0,m-1:N*m:m])
		solRKfast[1:N+1]=np.real(num_solStagesfast[0,m-1:N*m:m])

		errRK[j] = max(np.abs(solRK-ex_sol))
		errRKfast[j] = max(np.abs(solRKfast-ex_sol))
		errsCQ_vs_fastCQ[j] =np.sqrt(dt)*np.linalg.norm(np.abs(solRK-solRKfast))
		end = time.time()
		print(method," N: ", N, " Time: ", end-start)
	rk_results[method] = dict(Ns=count_classic_evals,counters=count_evals,err=errRK,errf=errRKfast,errfvsn=errsCQ_vs_fastCQ,freqs=np.array(freqs),tt=tt,solRK=solRK,solRKfast=solRKfast,m=m)
	prefix = method.replace("-","")
	res[prefix+"_Ns"]      = count_classic_evals
	res[prefix+"_counters"]= count_evals
	res[prefix+"_err"]     = errRK
	res[prefix+"_errf"]    = errRKfast
	res[prefix+"_errfvsn"] = errsCQ_vs_fastCQ

# ---------------------------------------------------------------------------
# Linear multistep methods (Implicit Euler, BDF-2, Trapezoidal rule), as in
# the scalar folder (scalar/cq_helpers.py): classic CQ vs. fast (parsimonious
# and Chebyshev-interpolated) CQ. The trapezoidal rule is only compared in
# its classic form, matching how it is used as a reference curve in the
# scalar folder's own convergence plots.
# ---------------------------------------------------------------------------
ms_methods = [
	("Implicit Euler", dict(lms=BDF1, method="BDF1", c=2, param=1, prefactL=1, fast=True, Am=Am_ie)),
	("BDF-2",          dict(lms=BDF2, method="BDF2", c=1, param=1, prefactL=1, fast=True)),
	("Trapezoidal",    dict(lms=TR,   prefactL=1,                              fast=False)),
]
ms_results = {}
for name,opts in ms_methods:
	Am_method = opts.get("Am",Am)
	Ns = np.zeros(Am_method)
	counters = np.zeros(Am_method)
	errCQ = np.zeros(Am_method)
	errfastCQ = np.zeros(Am_method)
	errsCQ_vs_fastCQ = np.zeros(Am_method)
	for j in range(Am_method):
		N=256*2**j
		dt = T/N
		Ns[j] = N
		tt=np.linspace(0,T,N+1)
		ex_sol = gd(tt)+gd(tt-1)+gd(tt-2)+gd(tt-3)+gd(tt-4)
		start = time.time()
		ucq = evalCQ(g,Ks,N,T,lms=opts["lms"],prefactL=opts["prefactL"])
		errCQ[j] = max(np.abs(ucq-ex_sol))
		if opts["fast"]:
			ufast,counterevals = evalfastCQ(g,Ks,N,T,lms=opts["lms"],c=opts["c"],param=opts["param"],prefactL=opts["prefactL"],method=opts["method"])
			counters[j] = counterevals
			errfastCQ[j] = max(np.abs(ufast-ex_sol))
			errsCQ_vs_fastCQ[j] = max(np.abs(ucq-ufast))
			sol_fast = ufast
		else:
			# No fast variant is used for the trapezoidal rule in the scalar
			# folder either; report the classic scheme on both curves.
			counters[j] = opts["prefactL"]*N
			errfastCQ[j] = errCQ[j]
			errsCQ_vs_fastCQ[j] = 0.0
			sol_fast = ucq
		end = time.time()
		print(name," N: ", N, " Time: ", end-start)
	ms_results[name] = dict(Ns=Ns,counters=counters,err=errCQ,errf=errfastCQ,errfvsn=errsCQ_vs_fastCQ,tt=tt,sol=ucq,solfast=sol_fast)
	prefix = name.replace(" ","")
	res[prefix+"_Ns"]      = Ns
	res[prefix+"_counters"]= counters
	res[prefix+"_err"]     = errCQ
	res[prefix+"_errf"]    = errfastCQ
	res[prefix+"_errfvsn"] = errsCQ_vs_fastCQ

res["taus"] = taus
scipy.io.savemat("../scalar/data/err_compare_HS_high_freq.mat", res)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# One fixed color per method; classic is always a straight line, fast is
# always dashed, so the curves stay visually separable.
method_colors = {
	"RadauIIA-2":     "tab:blue",
	"RadauIIA-3":     "tab:orange",
	"Implicit Euler": "tab:green",
	"BDF-2":          "tab:red",
	"Trapezoidal":    "tab:purple",
}
classic_linestyle = '-'
fast_linestyle    = '--'

plt.close('all')
plt.figure(5, figsize=(10, 6))
plt.rc('text', usetex=True)
plt.loglog(rk_results["RadauIIA-2"]["Ns"],taus**(2.0/3),linestyle='dotted',color='gray',label=r"$O(\tau^{2/3})$")
for method in rk_methods:
	r = rk_results[method]
	color = method_colors[method]
	plt.loglog(r["Ns"],r["err"],color=color,linestyle=classic_linestyle,label="Classic "+method)
	plt.loglog(r["counters"],r["errf"],color=color,linestyle=fast_linestyle,label="Fast "+method)
for name,opts in ms_methods:
	r = ms_results[name]
	color = method_colors[name]
	plt.loglog(r["Ns"],r["err"],color=color,linestyle=classic_linestyle,label="Classic "+name)
	if opts["fast"]:
		plt.loglog(r["counters"],r["errf"],color=color,linestyle=fast_linestyle,label="Fast "+name)
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel("Maximal error")
plt.title("Error caused by interpolation")
plt.grid(True)
plt.legend()
plt.savefig('errs_RK_high_freq.pdf')

for method in rk_methods:
	r = rk_results[method]
	plt.figure()
	freqs = r["freqs"]
	for j in range(r["m"]):
		plt.scatter(np.real(freqs[:,j]),np.imag(freqs[:,j]),label = "Stage "+str(j+1))
	plt.legend()
	plt.title("Frequencies - "+method)
	plt.savefig('freqs_RK_'+method.replace("-","")+'_high_freq.pdf')

plt.figure(13, figsize=(10, 6))
cut =-1
tt = rk_results["RadauIIA-2"]["tt"]
for method in rk_methods:
	r = rk_results[method]
	plt.plot(tt[:cut],r["solRKfast"][:cut],color=method_colors[method],linestyle=fast_linestyle,label="Fast "+method)
for name,opts in ms_methods:
	r = ms_results[name]
	linestyle = fast_linestyle if opts["fast"] else classic_linestyle
	plt.plot(r["tt"][:cut],r["solfast"][:cut],color=method_colors[name],linestyle=linestyle,label=name)
plt.plot(tt[:cut],(gd(tt)+gd(tt-1)+gd(tt-2)+gd(tt-3)+gd(tt-4))[:cut],linestyle='dotted',color='gray',label="Exact solution")
plt.xlabel("Time $t$")
plt.ylabel("Solution")
plt.grid(True)
plt.legend()
plt.title("Solution comparison")
plt.savefig('sol_RK_high_freq.pdf')
