import numpy as np
import matplotlib.pyplot as plt
from contour_helpers import frequencies_on_contour,graded_mesh,colored_line

## Which part of the contour zeta(l) = rho*exp(-2 pi i l) is mapped to which part of the
## eigenvalue branches s_1(l), s_2(l) of Delta(zeta)/tau (RadauIIA-2). Colour = Phi = pi*l.
## The eigenvalues are plotted scaled as tau*s, which is independent of tau.
N = 1024
T = 5
m = 2
lls,zeta,S,dt,rho = frequencies_on_contour(N,T)
L = len(lls)
Phi0 = graded_mesh(rho,m)[0]
Phi = np.pi*lls
tS = dt*S

plt.close('all')
plt.rc('text', usetex=True)
fig,axs = plt.subplots(2,2,figsize=(12,10))
cmap = 'viridis'
marks = [0,0.2,0.5,0.8]

## (a) Contour in the zeta-plane
ax = axs[0,0]
lc = colored_line(ax,np.real(zeta),np.imag(zeta),Phi,cmap,0,np.pi,lw=4)
ax.plot(np.cos(np.linspace(0,2*np.pi,400)),np.sin(np.linspace(0,2*np.pi,400)),color='gray',linestyle='dotted',lw=1)
for l in marks:
	j = int(round(l*L))
	ax.plot(np.real(zeta[j]),np.imag(zeta[j]),'ko',markersize=5)
	ax.annotate(r"$\Phi = %s$"%{0:"0",0.2:r"\pi/5",0.5:r"\pi/2",0.8:r"4\pi/5"}[l],(np.real(zeta[j]),np.imag(zeta[j])),
	            textcoords="offset points",xytext=(8,-14 if l<0.5 else 6))
j = int(0.06*L)
ax.annotate("",xy=(np.real(zeta[j+20]),np.imag(zeta[j+20])),xytext=(np.real(zeta[j]),np.imag(zeta[j])),
            arrowprops=dict(arrowstyle="->",lw=1.5))
ax.set_aspect('equal')
ax.set_xlim(-1.2,1.4)
ax.set_ylim(-1.2,1.2)
ax.set_xlabel(r"$\mathrm{Re}\,\zeta$")
ax.set_ylabel(r"$\mathrm{Im}\,\zeta$")
ax.set_title(r"Contour $\zeta = \rho\, e^{-2\pi i l}$, $\rho = %.4f$"%rho)
ax.grid(True)

## (b) Both branches in the (scaled) s-plane
ax = axs[0,1]
colored_line(ax,np.real(tS[0]),np.imag(tS[0]),Phi,cmap,0,np.pi,lw=4)
colored_line(ax,np.real(tS[1]),np.imag(tS[1]),Phi,cmap,0,np.pi,lw=1.5,ls='dashed')
ax.plot(np.real(tS[0,0]),np.imag(tS[0,0]),'ko',markersize=7,label=r"Branch 1 at $\Phi = 0$")
ax.plot(np.real(tS[1,0]),np.imag(tS[1,0]),'ks',markersize=7,label=r"Branch 2 at $\Phi = 0$")
for l in marks[1:]:
	j = int(round(l*L))
	for b,mk in [(0,'o'),(1,'s')]:
		ax.plot(np.real(tS[b,j]),np.imag(tS[b,j]),mk,color='k',markersize=4,markerfacecolor='none')
ax.text(0.15,-0.6,"Branch 1 (thick)")
ax.text(3.8,-2.4,"Branch 2 (dashed)")
ax.set_aspect('equal')
ax.set_xlabel(r"$\mathrm{Re}\,\tau s$")
ax.set_ylabel(r"$\mathrm{Im}\,\tau s$")
ax.set_title(r"Eigenvalues $\tau s_{1,2}(\zeta)$ of $\Delta(\zeta)$")
ax.legend(loc='lower left')
ax.grid(True)

## (c) Zoom of branch 1 near the imaginary axis, together with the poles of K
ax = axs[1,0]
I = Phi <= 1.15*np.pi/5
colored_line(ax,np.real(tS[0,I]),np.imag(tS[0,I]),Phi[I],cmap,0,np.pi,lw=3)
poles = dt*2*np.pi*np.arange(0,int(1.5/(dt*2*np.pi))+1)
ax.plot(np.zeros(len(poles)),poles,'x',color='C3',markersize=4,label=r"Poles $\tau s = 2\pi i k \tau$ of $K$")
for Ph,lab in [(Phi0,r"$\Phi_0$"),(np.pi/5,r"$\pi/5$")]:
	j = int(round(Ph/np.pi*L))
	ax.plot(np.real(tS[0,j]),np.imag(tS[0,j]),'ko',markersize=6)
	ax.annotate(lab,(np.real(tS[0,j]),np.imag(tS[0,j])),textcoords="offset points",xytext=(8,-4))
ax.set_xlim(-0.01,0.12)
ax.set_ylim(-0.05,1.5)
ax.set_xlabel(r"$\mathrm{Re}\,\tau s$")
ax.set_ylabel(r"$\mathrm{Im}\,\tau s$")
ax.set_title(r"Zoom: branch 1 for $\Phi \le 1.15\,\pi/5$")
ax.legend(loc='lower right')
ax.grid(True)

## (d) Distance to the imaginary axis (where the poles of K lie) along the contour
ax = axs[1,1]
I = lls <= 0.5
ax.semilogy(Phi[I],np.real(tS[0,I]),color='C0',lw=2,label=r"$\mathrm{Re}\,\tau s_1$")
ax.semilogy(Phi[I],np.real(tS[1,I]),color='C1',lw=2,linestyle='dashed',label=r"$\mathrm{Re}\,\tau s_2$")
ax.semilogy(Phi[I],np.abs(tS[0,I]),color='C0',lw=1,linestyle='dotted',label=r"$|\tau s_1|$")
ax.semilogy(Phi[I],np.abs(tS[1,I]),color='C1',lw=1,linestyle='dashdot',label=r"$|\tau s_2|$")
ax.axhline(2*np.pi*dt,color='C3',lw=1,label=r"Pole spacing $2\pi\tau$")
ax.axvspan(0,Phi0,color='gray',alpha=0.25,label=r"Exact evaluations, $\Phi < \Phi_0$")
ax.axvspan(Phi0,np.pi/5,color='C2',alpha=0.2,label=r"Graded mesh, $\Phi_0 \le \Phi < \pi/5$")
ax.set_xlim(0,np.pi/2)
ax.set_xlabel(r"$\Phi = \pi l$")
ax.set_title(r"Branches along the half contour $l \in [0,1/2]$")
ax.legend(fontsize=8,loc='lower right')
ax.grid(True)

cb = fig.colorbar(lc,ax=axs,shrink=0.6,location='right')
cb.set_label(r"$\Phi = \pi l$")
cb.set_ticks([0,np.pi/5,np.pi/2,4*np.pi/5,np.pi])
cb.set_ticklabels([r"$0$",r"$\pi/5$",r"$\pi/2$",r"$4\pi/5$",r"$\pi$"])
fig.suptitle(r"RadauIIA-2: contour vs.\ eigenvalue branches, $N = %d$"%N)
plt.savefig('contour_to_branches.pdf',bbox_inches='tight')
