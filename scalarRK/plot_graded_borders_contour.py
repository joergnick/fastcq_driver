import numpy as np
import matplotlib.pyplot as plt
from contour_helpers import contour_parameters,graded_mesh

## Borders Phi_j of the graded mesh drawn into the contour zeta(l) = rho*exp(-2 pi i l), with l = Phi/pi.
## The mirrored borders l = 1-Phi_j/pi (used for the conjugate branch) lie in the upper half plane.
## Usage: python3 plot_graded_borders_contour.py [RadauIIA-2|RadauIIA-3]
import sys
import scipy.io
method = sys.argv[1] if len(sys.argv) > 1 else "RadauIIA-2"
T = 5
if method == "RadauIIA-2":
	## as in convplots_RadauIIA2_graded.py
	N  = 8192
	m  = 2
	ct = 0.1
	tol = 1e-14
	code_borders = np.array([0.2,0.25,0.75,0.8])
	phi_switch = np.pi/5
	angle_labels = [(0.2,r"$\pi/5$"),(0.25,r"$\pi/4$"),(0.5,r"$\pi/2$")]
elif method == "RadauIIA-3":
	## as in convplots_RadauIIA3_graded.py
	N  = 4096
	m  = 3
	ct = 0.01
	tol = max(1e-14,0.01*(T*1.0/N)**4)
	code_borders = np.array([0.34,0.39,0.5,0.61,0.66])
	phi_switch = 0.34*np.pi
	angle_labels = [(0.34,r"$0.34\pi$"),(0.39,r"$0.39\pi$"),(0.5,r"$\pi/2$")]
L,dt,rho = contour_parameters(N,T,tol)
Phi = graded_mesh(rho,m,ct=ct,phi_switch=phi_switch)
zeta_of = lambda l: rho*np.exp(-1j*2*np.pi*l)
z_graded = zeta_of(Phi/np.pi)
z_mirror = zeta_of(1-Phi/np.pi)
z_code = zeta_of(code_borders)
z_grid = zeta_of(np.linspace(0,L-1,L)/L)

plt.close('all')
plt.rc('text', usetex=True)
fig,axs = plt.subplots(1,2,figsize=(15,7))
tt = np.linspace(0,1,2000)

def radial_ticks(ax,z,h,**kw):
	## Short radial segments through the points z (length 2h), as markers on the circle
	u = z/np.abs(z)
	segs = np.array([[np.real(z-h*u),np.real(z+h*u)],[np.imag(z-h*u),np.imag(z+h*u)]])
	lines = ax.plot(segs[0],segs[1],**kw)
	return lines[0]

k = min(40,len(Phi)-1)
h_zoom = 0.03*np.abs(z_graded[k]-z_graded[0])
for ax,zoom in zip(axs,[False,True]):
	h = h_zoom if zoom else 0.035
	ax.plot(np.real(zeta_of(tt)),np.imag(zeta_of(tt)),color='lightgray',lw=1.5,zorder=1)
	ax.plot(np.real(zeta_of(tt[tt < Phi[0]/np.pi])),np.imag(zeta_of(tt[tt < Phi[0]/np.pi])),color='gray',lw=4,alpha=0.6,zorder=1,
	        label=r"Exact evaluations, $\Phi < \Phi_0$")
	radial_ticks(ax,z_graded,h,color='C3',lw=0.8,zorder=3).set_label(r"Borders $\Phi_j$ of the graded mesh (%d pieces)"%(len(Phi)-1))
	radial_ticks(ax,z_mirror,h,color='C1',lw=0.8,alpha=0.6,zorder=3).set_label(r"Mirrored borders $l = 1-\Phi_j/\pi$ (branch 2)")
	ax.plot(np.real(z_code),np.imag(z_code),'D',color='k',markersize=6,zorder=4,label=r"Borders of the existing code, $l \in \{%s\}$"%(",".join(str(b) for b in code_borders)))
	ax.plot(np.real(z_graded[0]),np.imag(z_graded[0]),'o',color='C3',markersize=7,zorder=4)
	ax.set_aspect('equal')
	ax.set_xlabel(r"$\mathrm{Re}\,\zeta$")
	ax.set_ylabel(r"$\mathrm{Im}\,\zeta$")
	ax.grid(True)

ax = axs[0]
for l,lab in [(Phi[0]/np.pi,r"$\Phi_0$")]+angle_labels:
	z = zeta_of(l)
	ax.plot([0,np.real(z)],[0,np.imag(z)],color='k',lw=0.6,linestyle='dotted')
	ax.annotate(lab,(np.real(z),np.imag(z)),textcoords="offset points",xytext=(8,-12))
ax.set_xlim(-1.15,1.25)
ax.set_ylim(-1.15,1.15)
ax.set_title(r"Contour $\zeta = \rho e^{-2\pi i l}$, $\rho = %.5f$, $\Phi = \pi l$"%rho)
ax.legend(fontsize=8,loc='upper left')

## Zoom: start of the graded mesh, with the grid frequencies
ax = axs[1]
k = min(40,len(Phi)-1)
zz = z_graded[:k+1]
ax.plot(np.real(z_grid),np.imag(z_grid),'.',color='C0',markersize=3,zorder=2,label=r"Grid frequencies $l = j/L$")
zc = 0.5*(zz[0]+zz[-1])
r = 0.6*np.abs(zz[-1]-zz[0])
ax.set_xlim(np.real(zc)-r,np.real(zc)+r)
ax.set_ylim(np.imag(zc)-r,np.imag(zc)+r)
ax.annotate(r"$\Phi_0 = %.4f$"%Phi[0],(np.real(zz[0]),np.imag(zz[0])),textcoords="offset points",xytext=(10,4))
ax.annotate(r"$\Phi_{%d} = %.4f$"%(k,Phi[k]),(np.real(zz[k]),np.imag(zz[k])),textcoords="offset points",xytext=(10,4))
ax.set_title(r"Zoom: first %d borders $\Phi_0,\dots,\Phi_{%d}$"%(k+1,k))
ax.legend(fontsize=8,loc='lower right')

fig.suptitle(method+r": borders of the graded mesh on the contour, $N = %d$, $c_1 = 1$, $\widetilde c = %s$, $2m = %d$"%(N,ct,2*m))
plt.tight_layout()
plt.savefig('graded_borders_contour_'+method+'.pdf',bbox_inches='tight')
scipy.io.savemat("../scalar/data/graded_borders_"+method+".mat",dict(N=N,T=T,m=m,ct=ct,c1=1.0,tol=tol,rho=rho,L=L,Phi=Phi,
                 l_borders=Phi/np.pi,zeta_borders=z_graded,zeta_mirrored=z_mirror,code_borders=code_borders,zeta_code=z_code))
