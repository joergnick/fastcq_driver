import sys
sys.path.append('cqToolbox')

import numpy as np
from rkmethods import RKMethod
from tcheby_helpers import tchebycheff_nodes,tcheb_interpolation

## Helpers for the visualisation scripts plot_contour_to_branches.py, plot_frequency_pairs.py
## and plot_matrix_grading.py. The contour is zeta(l) = rho*exp(-2 pi i l), l = lls in [0,1),
## and the angle is measured as Phi = pi*l (Phi = pi/5 is the border cutoff_interp = 0.2).

Ks = lambda s: s/(1-np.exp(-s))

def contour_parameters(N,T=5,tol=1e-14):
	L   = 2*N
	dt  = T*1.0/N
	rho = tol**(1.0/(2*N))
	return L,dt,rho

def delta_matrix(zeta,method="RadauIIA-2"):
	rk = RKMethod(method,1)
	m  = rk.m
	return np.linalg.inv(rk.A+zeta*1.0/(1-zeta)*np.ones((m,1))*rk.b)

def frequencies_on_contour(N,T=5,tol=1e-14,method="RadauIIA-2"):
	## Eigenvalues of Delta(zeta)/dt on the contour, sorted into continuous branches
	## (as in Conv_Operator.apply_RKconvolfast)
	L,dt,rho = contour_parameters(N,T,tol)
	lls  = np.linspace(0,L-1,L)/L
	zeta = rho*np.exp(-1j*2*np.pi*lls)
	freqs, branches = [], []
	for j in range(L):
		deltaEigs = np.linalg.eigvals(delta_matrix(zeta[j],method)/dt)
		if j == 0:
			curr_branch = list(range(len(deltaEigs)))
		else:
			curr_branch = [int(np.argmin(np.abs(deltaEigs[i]-freqs[j-1][branches[-1]]))) for i in range(len(deltaEigs))]
		freqs.append(deltaEigs)
		branches.append(curr_branch)
	freqs = np.array(freqs)
	freqs_sorted = np.array([freqs[j,branches[j]] for j in range(L)]).T
	return lls,zeta,freqs_sorted,dt,rho

def graded_mesh(lam,m,c1=1.0,ct=0.1,phi_switch=np.pi/5):
	Phi = [c1*abs(np.log(lam))**(1.0/(2*m))]
	if Phi[0] >= phi_switch:
		return np.array([phi_switch])
	while Phi[-1] < phi_switch:
		if Phi[-1] < 1:
			Phi.append(Phi[-1]+ct*Phi[-1]**(2*m))
		else:
			Phi.append(np.pi)
	Phi[-1] = phi_switch
	return np.array(Phi)

def interpolate_on_pieces(f,lls,pieces,exact_vals):
	## Piecewise Chebyshev interpolation of f (values in the last axis-free shape of exact_vals[j])
	## on the grid lls. pieces = list of (a,b,n,is_graded); graded pieces with at most n grid points
	## are evaluated exactly. Returns the interpolated values, the used nodes and the exact points.
	vals = exact_vals.copy()
	nodes, exact_pts = [], []
	for (a,b,n,is_graded) in pieces:
		mask = (lls >= a) & (lls <= b)
		if is_graded and sum(mask) <= n:
			exact_pts.append(lls[mask])
			continue
		tch = np.sort(tchebycheff_nodes(a,b,n))
		fn  = np.array([f(x) for x in tch])
		nodes.append(tch)
		if sum(mask) > 0:
			x = 2*(lls[mask]-a)/(b-a)-1
			xn = 2*(tch-a)/(b-a)-1
			fn_flat = fn.reshape(len(tch),-1)
			interp = np.array([tcheb_interpolation(xn,fn_flat[:,k],x) for k in range(fn_flat.shape[1])]).T
			vals[mask] = interp.reshape((sum(mask),)+exact_vals.shape[1:])
	nodes = np.concatenate(nodes) if nodes else np.array([])
	exact_pts = np.concatenate(exact_pts) if exact_pts else np.array([])
	return vals,nodes,exact_pts

def colored_line(ax,x,y,c,cmap='viridis',vmin=0,vmax=1,lw=2,ls='solid',zorder=2):
	## Line through (x,y), coloured by the parameter c
	from matplotlib.collections import LineCollection
	import matplotlib.pyplot as plt
	points = np.array([x,y]).T.reshape(-1,1,2)
	segments = np.concatenate([points[:-1],points[1:]],axis=1)
	lc = LineCollection(segments,cmap=cmap,norm=plt.Normalize(vmin,vmax),linewidths=lw,linestyles=ls,zorder=zorder)
	lc.set_array(0.5*(c[:-1]+c[1:]))
	ax.add_collection(lc)
	return lc
