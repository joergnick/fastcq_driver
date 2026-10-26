import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from paper_style import ASTERISK,style_axes

## Comparison plot of the paper (cf. plotscripts/comparison_oneplot.m): classic and parsimonious
## implicit Euler, BDF-2, RadauIIA-2, RadauIIA-3 and the (classic) trapezoidal rule for the same
## problem (g = sin^2, K(s) = s/(1-exp(-s)), T = 5). Laplace evaluations are counted on one side of
## the real axis only for all schemes. Data from:
##   ../scalar/paperdata_LMS.py             -> ../scalar/data/paper_LMS_HS.mat
##   paperplot_RadauIIA2.py --recompute     -> ../scalar/data/paper_RadauIIA2_graded.mat
##   convplots_RadauIIA3_graded.py 0.0004   -> ../scalar/data/{err,sol}_RadauIIA3_graded_HS_ct0.0004.mat
## Two versions: "paper" uses the errors of the present paper figures (maximal error for the
## multistep methods and RadauIIA-2, weighted l^2 error for RadauIIA-3), "l2" the weighted l^2
## error for all schemes.
COLORS = ["#0072BD","#77AC30","#EDB120","#D95319","#7E2F8E"]

lms = scipy.io.loadmat("../scalar/data/paper_LMS_HS.mat")
r2  = scipy.io.loadmat("../scalar/data/paper_RadauIIA2_graded.mat")
r3  = scipy.io.loadmat("../scalar/data/err_RadauIIA3_graded_HS_ct0.0004.mat")
r3s = scipy.io.loadmat("../scalar/data/sol_RadauIIA3_graded_HS_ct0.0004.mat")
v = lambda d,k: d[k].ravel()

## RadauIIA-3: maximal errors from the saved solutions, N >= 32 as in the paper
N3 = np.round(5.0/v(r3,"taus")).astype(int)
I3 = N3 >= 32
sol3 = lambda N,k: r3s["N"+str(N)][k][0,0].ravel()
err3_max  = np.array([np.max(np.abs(sol3(N,"solRK")-sol3(N,"ex_sol"))) for N in N3[I3]])
errg3_max = np.array([np.max(np.abs(sol3(N,"solRKgraded")-sol3(N,"ex_sol"))) for N in N3[I3]])

def curves(norm):
	n3 = "l2" if norm in ["paper","l2"] else "max"
	nl = "max" if norm in ["paper","max"] else "l2"
	return [
		("Implicit Euler",v(lms,"L_classic"),v(lms,"err_BDF1_"+nl),v(lms,"L_BDF1f"),v(lms,"errf_BDF1_"+nl),COLORS[0]),
		("BDF-$2$",v(lms,"L_classic"),v(lms,"err_BDF2_"+nl),v(lms,"L_BDF2f"),v(lms,"errf_BDF2_"+nl),COLORS[1]),
		("RadauIIA-$2$",v(r2,"L_classic"),v(r2,"err_"+nl),v(r2,"L_graded"),v(r2,"errg_"+nl),COLORS[2]),
		("RadauIIA-$3$",v(r3,"Ns")[I3],v(r3,"err")[I3] if n3 == "l2" else err3_max,
		 v(r3,"counters_graded")[I3],v(r3,"errg")[I3] if n3 == "l2" else errg3_max,COLORS[3]),
	],("Trapezoidal rule",v(lms,"L_classic"),v(lms,"err_TR_"+nl),COLORS[4])

def comparison_figure(norm,filename):
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig,ax = plt.subplots(figsize=(6.5,4.6))
	cs,tr = curves(norm)
	allerr = []
	for (name,L,err,Lf,errf,col) in cs:
		ax.loglog(L,err,'-o',color=col,lw=1,markersize=6,markerfacecolor='none',label=name)
		ax.loglog(Lf,errf,'-',marker=ASTERISK,color=col,lw=1,markersize=6,label="Parsimonious "+name)
		allerr += [err,errf]
	name,L,err,col = tr
	ax.loglog(L,err,'-o',color=col,lw=1,markersize=6,markerfacecolor='none',label=name)
	allerr += [err]
	allerr = np.concatenate(allerr)
	ax.set_xlim(10**1,10**5.2)
	ax.set_ylim(min(10**(-5.2),0.7*allerr.min()),max(10**(-0.2),1.4*allerr.max()))
	ax.set_xlabel("Number of Laplace evaluations")
	ax.set_ylabel(r"Weighted $\ell^2-$ error")
	ax.set_title(r"\textsf{\textbf{Time discretization error}}",fontsize=11)
	ax.legend(loc='lower left',fontsize=7,fancybox=False,edgecolor='black',framealpha=1,borderpad=0.4,handlelength=2.5)
	style_axes(ax)
	ax.set_box_aspect(None)
	plt.tight_layout()
	plt.savefig(filename,bbox_inches='tight')

comparison_figure("paper","paper_comparison.pdf")
comparison_figure("l2","paper_comparison_l2.pdf")
