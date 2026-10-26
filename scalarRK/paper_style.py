import numpy as np
import matplotlib.pyplot as plt

## Figure style of the convergence plots in the paper (Figures 4 and 5): two boxed panels,
## classic method in MATLAB-blue with '*', parsimonious method in MATLAB-orange with open 'o',
## dashed black reference line, boxed legend, sans-serif titles and serif (LaTeX) axes.
BLUE   = (0.0,0.4470,0.7410)
ORANGE = (0.8500,0.3250,0.0980)
ASTERISK = (6,2,0)   ## MATLAB-like '*' marker

def decade_limits(*arrays):
	vals = np.concatenate([np.asarray(a,dtype=float).ravel() for a in arrays])
	vals = vals[vals > 0]
	return 10.0**np.floor(np.log10(vals.min())),10.0**np.ceil(np.log10(vals.max()))

def style_axes(ax):
	ax.minorticks_off()
	ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=9)
	for sp in ax.spines.values():
		sp.set_linewidth(0.5)
	ax.set_box_aspect(0.95)

def paper_figure(method,L_classic,err_classic,L_pars,err_pars,ref_order,ref_label,N_dev,dev,filename,ref_shift=1.0,
                 ref_anchor="first",pars_ref=None,xlim=None):
	## ref_shift: factor by which the reference line is moved away from the classic data point given
	## by ref_anchor ("first" or "last"); pars_ref = (order,label,shift): additional reference line
	## for the parsimonious method, anchored at its last data point (shifted by shift);
	## xlim: limits of the Laplace evaluation axis (default: enclosing decades)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10.5,4.6))
	legend_kw = dict(loc='upper right',fontsize=8,fancybox=False,edgecolor='black',framealpha=1,borderpad=0.4,handlelength=2.5)

	## Time discretization error vs. Laplace evaluations
	ax1.loglog(L_classic,err_classic,'-',marker=ASTERISK,color=BLUE,lw=1,markersize=6,label=method)
	ax1.loglog(L_pars,err_pars,'-o',color=ORANGE,lw=1,markersize=5,markerfacecolor='none',label="Parsimonious "+method)
	L_ref = np.asarray(L_classic,dtype=float)
	ia = 0 if ref_anchor == "first" else -1
	ax1.loglog(L_ref,ref_shift*err_classic[ia]*(L_ref/L_ref[ia])**(-ref_order),'--',color='black',lw=0.8,label=ref_label)
	if pars_ref is not None:
		order,label,shift = pars_ref
		L_p = np.asarray(L_pars,dtype=float)
		ax1.loglog(L_p,shift*err_pars[-1]*(L_p/L_p[-1])**(-order),'-.',color='black',lw=0.8,label=label)
	ax1.set_xlim(*(xlim if xlim is not None else decade_limits(L_classic,L_pars)))
	ax1.set_ylim(*decade_limits(err_classic,err_pars))
	ax1.set_xlabel("Number of Laplace evaluations")
	ax1.set_ylabel(r"Weighted $\ell^2-$ error")
	ax1.set_title(r"\textsf{\textbf{Time discretization error}}",fontsize=11)
	ax1.legend(**legend_kw)
	style_axes(ax1)

	## Error caused by interpolation vs. timesteps (zero deviations, i.e. exact evaluation, are not shown)
	N_dev = np.asarray(N_dev,dtype=float)
	dev = np.asarray(dev,dtype=float)
	I = dev > 0
	ax2.loglog(N_dev[I],dev[I],'-',marker=ASTERISK,color=BLUE,lw=1,markersize=6,label=r"$|K(\partial_t^\tau)g-I_p^\tau K(\partial_t^\tau)g|_2$")
	ax2.set_xlim(*decade_limits(N_dev[I]))
	ax2.set_ylim(*decade_limits(dev[I]))
	ax2.set_xlabel(r"Number of timesteps $N$")
	ax2.set_ylabel(r"Weighted $\ell^2-$ error")
	ax2.set_title(r"\textsf{\textbf{Error caused by interpolation}}",fontsize=11)
	ax2.legend(**legend_kw)
	style_axes(ax2)

	plt.tight_layout()
	plt.savefig(filename,bbox_inches='tight')
	return fig
