import numpy as np
import matplotlib.pyplot as plt
from contour_helpers import frequencies_on_contour,colored_line

## Which eigenvalues of Delta(zeta)/tau belong to the same contour point zeta(l) = rho*exp(-2 pi i l)
## (RadauIIA-2), how the two branches are connected, and the symmetry s_2(l) = conj(s_1(1-l)).
## The eigenvalues are plotted scaled as tau*s, which is independent of tau.
N = 1024
T = 5
lls,zeta,S,dt,rho = frequencies_on_contour(N,T)
L = len(lls)
Phi = np.pi*lls
tS = dt*S

plt.close('all')
plt.rc('text', usetex=True)
fig,axs = plt.subplots(1,3,figsize=(18,6.5))
cmap = plt.get_cmap('viridis')

## (a) Eigenvalue pairs belonging to the same zeta
ax = axs[0]
ax.plot(np.real(tS[0]),np.imag(tS[0]),color='lightgray',lw=1,zorder=1)
ax.plot(np.real(tS[1]),np.imag(tS[1]),color='lightgray',lw=1,zorder=1)
for l in np.arange(0,1,1.0/24):
	j = int(round(l*L))
	col = cmap(l)
	ax.plot(np.real(tS[:,j]),np.imag(tS[:,j]),color=col,lw=1)
	ax.plot(np.real(tS[0,j]),np.imag(tS[0,j]),'o',color=col,markersize=6)
	ax.plot(np.real(tS[1,j]),np.imag(tS[1,j]),'s',color=col,markersize=6)
ax.plot([],[],'ko',label=r"$\tau s_1(\zeta)$")
ax.plot([],[],'ks',label=r"$\tau s_2(\zeta)$")
ax.set_aspect('equal')
ax.set_xlabel(r"$\mathrm{Re}\,\tau s$")
ax.set_ylabel(r"$\mathrm{Im}\,\tau s$")
ax.set_title(r"Pairs $(\tau s_1(\zeta),\tau s_2(\zeta))$ for $l = k/24$")
ax.legend(loc='lower left')
ax.grid(True)
sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,np.pi))
cb = fig.colorbar(sm,ax=ax,shrink=0.7,location='bottom',pad=0.12)
cb.set_label(r"$\Phi = \pi l$")
cb.set_ticks([0,np.pi/5,np.pi/2,4*np.pi/5,np.pi])
cb.set_ticklabels([r"$0$",r"$\pi/5$",r"$\pi/2$",r"$4\pi/5$",r"$\pi$"])

## (b) Following branch 1 once around the contour ends at the start of branch 2 and vice versa:
## both branches form a single closed curve, which is run through when zeta circles twice.
ax = axs[1]
loop = np.concatenate((tS[0],tS[1],tS[0,:1]))
tpar = np.concatenate((lls,1+lls,[2.0]))
lc = colored_line(ax,np.real(loop),np.imag(loop),tpar,'plasma',0,2,lw=3)
for t in [0.1,0.35,0.65,0.9]:
	for b in [0,1]:
		j = int(round(t*L))
		ax.annotate("",xy=(np.real(tS[b,j+8]),np.imag(tS[b,j+8])),xytext=(np.real(tS[b,j]),np.imag(tS[b,j])),
		            arrowprops=dict(arrowstyle="-|>",lw=1.5,color='k'),zorder=3)
ax.plot(np.real(tS[0,0]),np.imag(tS[0,0]),'ko',markersize=8)
ax.annotate(r"$s_1(0) = s_2(1^-)$",(np.real(tS[0,0]),np.imag(tS[0,0])),textcoords="offset points",xytext=(10,-4))
ax.plot(np.real(tS[1,0]),np.imag(tS[1,0]),'ks',markersize=8)
ax.annotate(r"$s_2(0) = s_1(1^-)$",(np.real(tS[1,0]),np.imag(tS[1,0])),textcoords="offset points",xytext=(-40,12))
ax.set_aspect('equal')
ax.set_xlabel(r"$\mathrm{Re}\,\tau s$")
ax.set_ylabel(r"$\mathrm{Im}\,\tau s$")
ax.set_title(r"One closed curve: $s_1$ for $l\in[0,1)$, then $s_2$")
ax.grid(True)
cb = fig.colorbar(lc,ax=ax,shrink=0.7,location='bottom',pad=0.12)
cb.set_label(r"Loop parameter: $l$ on branch 1, $1+l$ on branch 2")

## (c) Mirror symmetry over the real axis: s_2(l) = conj(s_1(1-l))
ax = axs[2]
ax.plot(np.real(tS[0]),np.imag(tS[0]),color='C0',lw=3,alpha=0.5,label=r"$\tau s_1(l)$")
ax.plot(np.real(tS[1]),np.imag(tS[1]),color='C1',lw=3,alpha=0.5,label=r"$\tau s_2(l)$")
idx = (L-np.arange(L)) % L
ax.plot(np.real(tS[0,idx])[1::40],-np.imag(tS[0,idx])[1::40],'k.',markersize=4,label=r"$\overline{\tau s_1(1-l)}$")
for l in [0.05,0.15,0.3,0.45]:
	j = int(round(l*L))
	a = tS[0,L-j]
	ax.annotate("",xy=(np.real(a),-np.imag(a)),xytext=(np.real(a),np.imag(a)),
	            arrowprops=dict(arrowstyle="<->",lw=1,linestyle='dotted',color='gray'))
	ax.plot(np.real(a),np.imag(a),'o',color='C0',markersize=6)
	ax.plot(np.real(tS[1,j]),np.imag(tS[1,j]),'s',color='C1',markersize=6)
	ax.annotate(r"$l = %.2f$"%l,(np.real(tS[1,j]),np.imag(tS[1,j])),textcoords="offset points",xytext=(6,-12),fontsize=8)
ax.axhline(0,color='k',lw=0.8)
ax.set_aspect('equal')
ax.set_xlabel(r"$\mathrm{Re}\,\tau s$")
ax.set_ylabel(r"$\mathrm{Im}\,\tau s$")
ax.set_title(r"Symmetry $s_2(l) = \overline{s_1(1-l)}$: branch 2 on $[0,\frac12]$ from branch 1 on $[\frac12,1]$")
ax.legend(loc='center')
ax.grid(True)

fig.suptitle(r"RadauIIA-2: eigenvalues belonging to the same frequency, $N = %d$"%N)
plt.savefig('frequency_pairs.pdf',bbox_inches='tight')
