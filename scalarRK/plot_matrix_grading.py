import numpy as np
import matplotlib.pyplot as plt
from contour_helpers import Ks,contour_parameters,delta_matrix,frequencies_on_contour,graded_mesh,interpolate_on_pieces

## Grading on the matrix level (RadauIIA-2): the graded mesh is placed on the contour
## zeta(l) = rho*exp(-2 pi i l), l in [0,1/2] (the rest follows by conjugation), and at every node the
## full matrix K(Delta(zeta)/tau) = V diag(K(s_1),K(s_2)) V^{-1} is evaluated and its entries are
## interpolated. Every node therefore costs m = 2 Laplace evaluations, but no eigenvalue branches
## have to be tracked. Compared to the grading on the eigenvalue level (branch 1 on [0,1] graded,
## branch 2 from s_2(l) = conj(s_1(1-l))).
N  = 8192
T  = 5
m  = 2
ct = 0.1
n_graded = 5
lls,zeta,S,dt,rho = frequencies_on_contour(N,T)
L = len(lls)
d = int(10*np.log(N)+10)
g = graded_mesh(rho,m,ct=ct)/np.pi
graded = [(g[k],g[k+1],n_graded,True) for k in range(len(g)-1)]
H = lls <= 0.5

def zeta_of(x):
	return rho*np.exp(-1j*2*np.pi*x)

def Kmat(x):
	deltaEigs,V = np.linalg.eig(delta_matrix(zeta_of(x))/dt)
	return V.dot(np.diag(Ks(deltaEigs))).dot(np.linalg.inv(V))

## Matrix level: l in [0,1/2]
M_ex = np.array([Kmat(x) for x in lls[H]])
pieces_M = graded+[(g[-1],0.25,d,False),(0.25,0.5,d,False)]
M_int,nodes_M,exact_M = interpolate_on_pieces(Kmat,lls[H],pieces_M,M_ex)
exact_M = np.concatenate((lls[lls < g[0]],exact_M))
count_M = m*(len(exact_M)+len(nodes_M))

## Eigenvalue level: branch 1 on [0,1], branch 2 by conjugation
def K1(x):
	deltaEigs = np.linalg.eigvals(delta_matrix(zeta_of(x))/dt)
	jn = min(int(round(x*L)),L-1)
	return Ks(deltaEigs[np.argmin(np.abs(deltaEigs-S[0,jn]))])
K1_ex = Ks(S[0])
K2_ex = Ks(S[1])
pieces_E = graded+[(g[-1],0.25,d,False),(0.25,1,d,False)]
K1_int,nodes_E,exact_E = interpolate_on_pieces(K1,lls,pieces_E,K1_ex)
exact_E = np.concatenate((lls[lls < g[0]],exact_E))
K2_int = K2_ex.copy()
K2_int[1:] = np.conj(K1_int[L-np.arange(1,L)])
count_E = len(exact_E)+len(nodes_E)+1
print("N: ",N," Laplace evaluations matrix level: ",count_M," eigenvalue level: ",count_E)

## Counts for several N (without evaluating K)
def counts(N,ct):
	L,dt,rho = contour_parameters(N,T)
	lls = np.linspace(0,L-1,L)/L
	d = int(10*np.log(N)+10)
	g = graded_mesh(rho,m,ct=ct)/np.pi
	graded = sum(min(n_graded,sum((lls >= g[k]) & (lls <= g[k+1]))) for k in range(len(g)-1))
	exact = sum(lls < g[0])
	classic = N*m
	fast = (m*sum(lls <= 0.2)+4*d)/2.0
	eig_level = exact+graded+2*d+1
	mat_level = m*(exact+graded+2*d)
	return classic,fast,eig_level,mat_level
Ns = 64*2**np.arange(10)

plt.close('all')
plt.rc('text', usetex=True)
fig,axs = plt.subplots(2,3,figsize=(19,11.5))
c_exact,c_graded,c_cheb = 'gray','C3','C2'
def split_nodes(nodes):
	return nodes[nodes < g[-1]],nodes[nodes >= g[-1]]
graded_M,cheb_M = split_nodes(nodes_M)

## (a) Matrix-level nodes on the contour
ax = axs[0,0]
tt = np.linspace(0,1,600)
ax.plot(np.real(zeta_of(tt)),np.imag(zeta_of(tt)),color='lightgray',lw=1)
ax.plot(np.real(zeta_of(exact_M)),np.imag(zeta_of(exact_M)),'.',color=c_exact,markersize=3,label="Exact evaluation (%d points)"%len(exact_M))
ax.plot(np.real(zeta_of(graded_M)),np.imag(zeta_of(graded_M)),'x',color=c_graded,markersize=4,label="Nodes on graded pieces (%d)"%len(graded_M))
ax.plot(np.real(zeta_of(cheb_M)),np.imag(zeta_of(cheb_M)),'o',color=c_cheb,markersize=3,label="Chebyshev nodes, $\Phi \ge \pi/5$ (%d)"%len(cheb_M))
for Ph,lab in [(g[0]*np.pi,r"$\Phi_0$"),(np.pi/5,r"$\pi/5$"),(np.pi/2,r"$\pi/2$")]:
	z = zeta_of(Ph/np.pi)
	ax.plot([0,np.real(z)],[0,np.imag(z)],color='k',lw=0.6,linestyle='dotted')
	ax.annotate(lab,(np.real(z),np.imag(z)),textcoords="offset points",xytext=(6,-10))
ax.set_aspect('equal')
ax.set_xlabel(r"$\mathrm{Re}\,\zeta$")
ax.set_ylabel(r"$\mathrm{Im}\,\zeta$")
ax.set_title(r"Matrix level: nodes on the contour, $l \in [0,\frac12]$")
ax.legend(fontsize=8,loc='upper left')
ax.grid(True)

## (b) Images of the matrix-level nodes: every zeta-node lands on both branches
ax = axs[0,1]
ax.plot(np.real(dt*S[0]),np.imag(dt*S[0]),color='C0',lw=1,alpha=0.4,label="Branch 1")
ax.plot(np.real(dt*S[1]),np.imag(dt*S[1]),color='C1',lw=1,alpha=0.4,label="Branch 2")
def eig_images(xs):
	return np.array([np.linalg.eigvals(delta_matrix(zeta_of(x))) for x in xs]).reshape(-1)
for xs,col,mk in [(exact_M,c_exact,'.'),(graded_M,c_graded,'x'),(cheb_M,c_cheb,'o')]:
	im = eig_images(xs)
	ax.plot(np.real(im),np.imag(im),mk,color=col,markersize=3)
ax.annotate("graded nodes on\nbranch 2 (harmless)",xy=(5.2,-2.2),xytext=(2.2,-1.8),fontsize=9,arrowprops=dict(arrowstyle="->"))
ax.annotate("graded nodes on\nbranch 1 (near poles)",xy=(0.05,1.1),xytext=(0.8,1.5),fontsize=9,arrowprops=dict(arrowstyle="->"))
ax.set_aspect('equal')
ax.set_xlabel(r"$\mathrm{Re}\,\tau s$")
ax.set_ylabel(r"$\mathrm{Im}\,\tau s$")
ax.set_title(r"Images of the nodes: both eigenvalues $\tau s_{1,2}(\zeta)$")
ax.legend(fontsize=8,loc='lower left')
ax.grid(True)

## (c) Size of the oscillating part K(s)-s = s e^{-s}/(1-e^{-s}) on both branches
ax = axs[0,2]
osc = lambda s: np.abs(s*np.exp(-s)/(1-np.exp(-s)))
ax.semilogy(np.pi*lls[H],osc(S[0,H]),color='C0',lw=1.5,label=r"Branch 1, $|K(s_1)-s_1|$")
osc2 = np.maximum(osc(S[1,H]),1e-300)
ax.semilogy(np.pi*lls[H],osc2,color='C1',lw=1.5,label=r"Branch 2, $|K(s_2)-s_2|$ ($<10^{-300}$, clipped)")
ax.semilogy(np.pi*lls[H],np.abs(S[0,H]),color='C0',lw=1,linestyle='dotted',label=r"$|s_1|$")
ax.axvspan(0,g[0]*np.pi,color='gray',alpha=0.25,label=r"Exact, $\Phi < \Phi_0$")
ax.axvspan(g[0]*np.pi,np.pi/5,color='C3',alpha=0.1,label="Graded region")
ax.set_xlim(0,np.pi/2)
ax.set_ylim(1e-12,1e5)
ax.set_xlabel(r"$\Phi = \pi l$")
ax.set_title(r"Oscillating part of $K$: only branch 1 near $\Phi = 0$ matters")
ax.legend(fontsize=8,loc='lower left')
ax.grid(True)

## (d) Zoom on a few graded pieces: oscillating parts of K(Delta/tau)_{11} and K(s_1)
ax = axs[1,0]
a,b = 0.4290,0.4405
def Dmat(x):
	return delta_matrix(zeta_of(x))/dt
def K1s(x):
	deltaEigs = np.linalg.eigvals(Dmat(x))
	jn = min(int(round(x*L)),L-1)
	return deltaEigs[np.argmin(np.abs(deltaEigs-S[0,jn]))]
xf = np.linspace(a,b,1500)/np.pi
ax.plot(np.pi*xf,np.real([Kmat(x)[0,0]-Dmat(x)[0,0] for x in xf]),color='C0',lw=1,label=r"$\mathrm{Re}\,(K(\Delta/\tau)-\Delta/\tau)_{11}$")
ax.plot(np.pi*xf,np.real([K1(x)-K1s(x) for x in xf]),color='C1',lw=1,label=r"$\mathrm{Re}\,(K(s_1)-s_1)$")
xg = lls[H][(np.pi*lls[H] >= a) & (np.pi*lls[H] <= b)]
J = (np.pi*lls[H] >= a) & (np.pi*lls[H] <= b)
ax.plot(np.pi*xg,np.real(M_int[J,0,0]-np.array([Dmat(x)[0,0] for x in xg])),'.',color='C0',markersize=6,label="Interpolants on the grid")
J1 = (np.pi*lls >= a) & (np.pi*lls <= b)
ax.plot(np.pi*lls[J1],np.real(K1_int[J1]-S[0,J1]),'.',color='C1',markersize=6)
for (pa,pb,n,is_graded) in graded:
	if pb*np.pi >= a and pa*np.pi <= b:
		ax.axvline(np.pi*pa,color='gray',lw=0.6,linestyle='dotted')
nd = nodes_M[(np.pi*nodes_M >= a) & (np.pi*nodes_M <= b)]
ax.plot(np.pi*nd,np.real([Kmat(x)[0,0]-Dmat(x)[0,0] for x in nd]),'x',color=c_graded,markersize=7,label="Nodes")
ax.plot(np.pi*nd,np.real([K1(x)-K1s(x) for x in nd]),'x',color=c_graded,markersize=7)
ax.set_xlim(a,b)
ax.set_xlabel(r"$\Phi = \pi l$")
ax.set_title(r"Graded pieces (dotted borders) at $\Phi \approx 0.43$: oscillating parts")
ax.legend(fontsize=8,loc='upper right')
ax.grid(True)

## (e) Interpolation errors along the half contour
ax = axs[1,1]
err_M = np.max(np.abs(M_int-M_ex).reshape(len(M_ex),-1),axis=1)
ax.semilogy(np.pi*lls[H],err_M,color='C0',lw=1,label=r"Matrix level, $\max_{ik}|K(\Delta/\tau)_{ik}-\tilde K_{ik}|$")
ax.semilogy(np.pi*lls[H],np.abs(K1_int-K1_ex)[H],color='C1',lw=1,alpha=0.8,label=r"Eigenvalue level, branch 1")
ax.semilogy(np.pi*lls[H],np.abs(K2_int-K2_ex)[H],color='C2',lw=1,alpha=0.8,label=r"Eigenvalue level, branch 2")
ax.axvspan(g[0]*np.pi,np.pi/5,color='C3',alpha=0.1,label="Graded region")
ax.set_xlim(0,np.pi/2)
ax.set_xlabel(r"$\Phi = \pi l$")
ax.set_ylabel("Interpolation error")
ax.set_title(r"Interpolation errors, $N = %d$, $\widetilde c = %s$"%(N,ct))
ax.legend(fontsize=8,loc='lower right')
ax.grid(True)

## (f) Laplace evaluations
ax = axs[1,2]
for c,ls in [(0.1,'solid'),(0.01,'dashed')]:
	C = np.array([counts(n,c) for n in Ns])
	if c == 0.1:
		ax.loglog(Ns,C[:,0],color='C0',marker='s',label="Classic")
		ax.loglog(Ns,C[:,1],color='C1',marker='o',label="Modified (existing)")
	ax.loglog(Ns,C[:,2],color='C4',marker='d',linestyle=ls,label=r"Eigenvalue level, $\widetilde c = %s$"%c)
	ax.loglog(Ns,C[:,3],color='C3',marker='x',linestyle=ls,label=r"Matrix level, $\widetilde c = %s$"%c)
ax.set_xlabel("Number of time steps $N$")
ax.set_ylabel("Laplace evaluations (one side of the real axis)")
ax.set_title("Cost of the gradings")
ax.legend(fontsize=8)
ax.grid(True)

fig.suptitle(r"RadauIIA-2: grading on the matrix level vs.\ on the eigenvalue level, $N = %d$, $\widetilde c = %s$"%(N,ct))
plt.tight_layout()
plt.savefig('matrix_grading.pdf',bbox_inches='tight')
