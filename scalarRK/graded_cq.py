import sys
sys.path.append('cqToolbox')

import numpy as np
from rkmethods import RKMethod
from tcheby_helpers import tchebycheff_nodes,tcheb_interpolation
from contour_helpers import interpolate_on_pieces

## Graded variants of Conv_Operator.apply_RKconvolfast, used by convplots_RadauIIA2_graded.py,
## convplots_RadauIIA3_graded.py and the paper figure scripts paperplot_RadauIIA{2,3}.py.

## Graded variant of Conv_Operator.apply_RKconvolfast for RadauIIA-2.
## The angles lls in [0,1) of the contour points rho*exp(-2 pi i lls) are measured as Phi = pi*lls,
## such that the lowest border cutoff_interp = 0.2 of the existing code corresponds to Phi = pi/5.
## Near Phi = 0 the exact evaluations of the existing code are replaced by the graded mesh
##   Phi_0     = c_1 |log(lambda)|^(1/(2m)),
##   Phi_{j+1} = Phi_j + ct*Phi_j^(2m)  (if Phi_j < 1, else pi),
## with lambda = rho. Below Phi_0 K is evaluated exactly, on each graded piece [Phi_j,Phi_{j+1}]
## K is interpolated in n_graded Chebyshev nodes (or evaluated exactly, if the piece contains
## at most n_graded frequencies). Once Phi_j >= pi/5, the pieces of the existing code are used.

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

def apply_RKconvolfast_graded(op,rhs,T,K,method="RadauIIA-2",c1=1.0,ct=0.1,n_graded=5,same_grading=False,first_value_is_t0=False):
	m = RKMethod(method,1).m
	if m != 2:
		raise ValueError("Graded variant only implemented for RadauIIA-2.")
	op.m = m
	[rhs,N,flag_1Dinput] = op.format_rhs(rhs,m,first_value_is_t0)
	[A_RK,b_RK,c_RK,m] = op.get_method_characteristics(method)
	L,dt,tol,rho = op.get_integration_parameters(N,T)
	rhs_fft = 1j*np.zeros((len(rhs[:,0]),m*L))
	for stageInd in range(m):
		rhs_fft[:,stageInd:m*L:m] = op.scale_fft(rhs[:,stageInd:m*N:m],N,T)

	def stage_freqs(zeta):
		deltaMatrix = np.linalg.inv(A_RK+zeta*1.0/(1-zeta)*np.ones((m,1))*b_RK)
		return np.linalg.eig(deltaMatrix/dt)

	## Frequencies on the contour and branch tracking, as in apply_RKconvolfast
	lls = np.linspace(0,L-1,L)/L
	s_vect = rho*np.exp(-1j*2*np.pi*lls)
	freqs, Ts, branches = [], [], []
	for j in range(L):
		deltaEigs,Tj = stage_freqs(s_vect[j])
		if j == 0:
			curr_branch = list(range(m))
		else:
			curr_branch = [int(np.argmin(np.abs(deltaEigs[stageInd]-freqs[j-1][branches[-1]]))) for stageInd in range(m)]
		freqs.append(deltaEigs)
		Ts.append(Tj)
		branches.append(curr_branch)
	freqs = np.array(freqs)
	Kvals = K(freqs)
	freqs_sorted = np.array([freqs[j,branches[j]] for j in range(L)]).T
	Kvals_sorted = np.array([Kvals[j,branches[j]] for j in range(L)]).T

	## Pieces: graded mesh up to pi/5, then the borders of apply_RKconvolfast (branch 1 mirrored)
	d = int(10*np.log(N)+10)
	g = graded_mesh(rho,m,c1,ct)/np.pi
	graded0 = [(g[k],g[k+1]) for k in range(len(g)-1)]
	graded1 = [(1-b,1-a) for (a,b) in graded0]
	pieces = [[(a,b,n_graded,True) for (a,b) in graded0]+[(g[-1],0.25,d,False),(0.25,1,d,False)],
	          [(a,b,n_graded,True) for (a,b) in graded1]+[(0,0.75,d,False),(0.75,1-g[-1],d,False)]]
	## Same grading for all branches: the frequencies of branch 2 are the mirrored ones of branch 1,
	## s_2(l) = conj(s_1(1-l)) for l > 0, hence K on branch 2 is obtained from branch 1 without
	## further evaluations (except for l = 0, where both frequencies are real)
	if same_grading:
		pieces[1] = []
		count_evals = sum(lls < g[0])+1
	else:
		count_evals = sum(lls < g[0])+sum(lls > 1-g[0])
	Kvals_interp_sorted = Kvals_sorted.copy()
	for si in range(m):
		for (a,b,n,is_graded) in pieces[si]:
			mask = (lls >= a) & (lls <= b)
			if is_graded and sum(mask) <= n:
				count_evals += sum(mask)
				continue
			tch = np.sort(tchebycheff_nodes(a,b,n))
			Kn = []
			for x in tch:
				deltaEigs,_ = stage_freqs(rho*np.exp(-1j*2*np.pi*x))
				jn = min(int(round(x*L)),L-1)
				idx = np.argmin(np.abs(deltaEigs-freqs_sorted[si,jn]))
				Kn.append(K(deltaEigs[idx]))
				count_evals += 1
			if sum(mask) > 0:
				Kvals_interp_sorted[si,mask] = tcheb_interpolation(2*(tch-a)/(b-a)-1,np.array(Kn),2*(lls[mask]-a)/(b-a)-1)
	if same_grading:
		Kvals_interp_sorted[1,1:] = np.conj(Kvals_interp_sorted[0,L-np.arange(1,L)])
	## Exact evaluation, if it is not more expensive (with same grading, evaluations are counted on one side only)
	max_count = L*m/2 if same_grading else L*m
	if count_evals >= max_count:
		Kvals_interp_sorted = Kvals_sorted
		count_evals = max_count
	Kvals_interp = np.zeros((L,m),dtype=complex)
	for j in range(L):
		Kvals_interp[j,branches[j]] = Kvals_interp_sorted[:,j]

	## Application of K in the diagonalised frequency domain
	HalfL = int(np.ceil(float(L)/2.0))
	phi_hat = 1j*np.zeros((len(rhs[:,0]),m*L))
	for j in range(0,HalfL+1):
		Tj = Ts[j]
		rhsStages = np.matmul(rhs_fft[:,m*j:m*(j+1)],np.linalg.inv(Tj).T)
		lhsStages = Kvals_interp[j,:]*rhsStages
		phi_hat[:,m*j:m*(j+1)] = np.matmul(lhsStages,Tj.T)
	for freqInd in range(HalfL,L):
		for stageInd in range(m):
			phi_hat[:,freqInd*m+stageInd] = np.conj(phi_hat[:,m*(L-freqInd)+stageInd])
	for stageInd in range(m):
		phi_hat[:,stageInd:m*L:m] = op.rescale_ifft(phi_hat[:,stageInd:m*L:m],N,T)
	phi_sol = phi_hat[:,:m*N]
	if first_value_is_t0:
		phi_sol = np.concatenate((np.zeros((len(phi_sol[:,0]),1)),phi_sol),axis=1)
	if flag_1Dinput:
		phi_sol = phi_sol[0,:]
	info = dict(Phi=g*np.pi,lls=lls,Kerr=np.abs(Kvals_interp_sorted-Kvals_sorted))
	return phi_sol,count_evals,info

## Graded variant of Conv_Operator.apply_RKconvolfast for RadauIIA-3 (cf. convplots_RadauIIA2_graded.py).
## Angles are measured as Phi = pi*l on the contour rho*exp(-2 pi i l); the lowest border
## cutoff_interp = 0.34 of the existing code corresponds to Phi = 0.34*pi.
## Branches (sorted by |s| at l = 0, tracked as a branch -> eigenvalue map, since the monodromy
## is a 3-cycle b_0 -> b_1 -> b_2 -> b_0):
##   b_0: small eigenvalue near l = 0, close to the poles of K -> graded on [Phi_0, 0.34*pi],
##        Phi_0 = c_1 |log(rho)|^(1/(2m)), Phi_{j+1} = Phi_j + ct*Phi_j^(2m), then the pieces
##        [0.34,0.39], [0.39,0.5], [0.5,1] of the existing code.
##   b_1: s_1(l) = conj(s_1(1-l)), harmless -> piece [0,1] of the existing code; only the Chebyshev
##        nodes with l <= 1/2 are evaluated, the others follow by conjugation.
##   b_2: s_2(l) = conj(s_0(1-l)) -> obtained from b_0 without further evaluations.
## Laplace evaluations are counted on one side of the real axis for all schemes.

def track_branches(zeta,dt,A_RK,b_RK,m):
	freqs, Ts, perms = [], [], []
	for j in range(len(zeta)):
		deltaMatrix = np.linalg.inv(A_RK+zeta[j]*1.0/(1-zeta[j])*np.ones((m,1))*b_RK)
		deltaEigs,Tj = np.linalg.eig(deltaMatrix/dt)
		if j == 0:
			## smallest eigenvalue first, then the conjugate pair ordered by decreasing imaginary part
			i0 = int(np.argmin(np.abs(deltaEigs)))
			rest = [i for i in range(m) if i != i0]
			perm = [i0]+sorted(rest,key=lambda i: -np.imag(deltaEigs[i]))
		else:
			perm = [int(np.argmin(np.abs(deltaEigs-p))) for p in freqs[-1]]
			if len(set(perm)) < m:
				raise ValueError("Branch tracking failed at j = "+str(j))
		freqs.append(deltaEigs[perm])
		Ts.append((deltaEigs,Tj))
		perms.append(perm)
	return np.array(freqs).T,Ts,perms

def apply_RKconvolfast_graded3(op,rhs,T,K,c1=1.0,ct=0.01,n_graded=5,first_value_is_t0=False):
	method = "RadauIIA-3"
	m = RKMethod(method,1).m
	op.m = m
	[rhs,N,flag_1Dinput] = op.format_rhs(rhs,m,first_value_is_t0)
	[A_RK,b_RK,c_RK,m] = op.get_method_characteristics(method)
	L,dt,tol,rho = op.get_integration_parameters(N,T)
	rhs_fft = 1j*np.zeros((len(rhs[:,0]),m*L))
	for stageInd in range(m):
		rhs_fft[:,stageInd:m*L:m] = op.scale_fft(rhs[:,stageInd:m*N:m],N,T)

	lls = np.linspace(0,L-1,L)/L
	zeta_of = lambda x: rho*np.exp(-1j*2*np.pi*x)
	S,Ts,perms = track_branches(zeta_of(lls),dt,A_RK,b_RK,m)
	K_ex = K(S)

	def K_branch(b):
		def f(x):
			z = zeta_of(x)
			deltaEigs = np.linalg.eigvals(np.linalg.inv(A_RK+z*1.0/(1-z)*np.ones((m,1))*b_RK)/dt)
			jn = min(int(round(x*L)),L-1)
			return K(deltaEigs[np.argmin(np.abs(deltaEigs-S[b,jn]))])
		return f

	d = int(15*np.log(N)-30)
	g = graded_mesh(rho,m,c1,ct,phi_switch=0.34*np.pi)/np.pi
	pieces0 = [(g[k],g[k+1],n_graded,True) for k in range(len(g)-1)]+[(g[-1],0.39,d,False),(0.39,0.5,d,False),(0.5,1,d,False)]
	## Once Phi_j >= 1 the recursion jumps to pi, i.e. the last graded piece is [Phi_j,0.34] with
	## Phi_j >= 1; it is not resolved by the grading and treated like the pieces of the existing code
	if len(g) > 1 and g[-2]*np.pi >= 1:
		pieces0[len(g)-2] = (g[-2],g[-1],d,False)
	K_int = K_ex.copy()
	K_int[0],nodes0,exact0 = interpolate_on_pieces(K_branch(0),lls,pieces0,K_ex[0])
	count_evals = sum(lls < g[0])+len(exact0)+len(nodes0)

	## Branch 1: Chebyshev nodes on [0,1] are symmetric about 1/2
	f1 = K_branch(1)
	cache = dict()
	def f1_sym(x):
		key = round(min(x,1-x),14)
		if key not in cache:
			cache[key] = f1(key)
		return cache[key] if x <= 0.5 else np.conj(cache[key])
	K_int[1],nodes1,_ = interpolate_on_pieces(f1_sym,lls,[(0,1,d,False)],K_ex[1])
	count_evals += len(cache)

	## Branch 2 from branch 0 by conjugation (l = 0 evaluated exactly)
	K_int[2,1:] = np.conj(K_int[0,L-np.arange(1,L)])
	count_evals += 1
	if count_evals >= L*m/2:
		K_int = K_ex
		count_evals = L*m/2

	## Application of K in the diagonalised frequency domain
	HalfL = int(np.ceil(float(L)/2.0))
	phi_hat = 1j*np.zeros((len(rhs[:,0]),m*L))
	for j in range(0,HalfL+1):
		deltaEigs,Tj = Ts[j]
		Kj = np.zeros(m,dtype=complex)
		Kj[perms[j]] = K_int[:,j]
		rhsStages = np.matmul(rhs_fft[:,m*j:m*(j+1)],np.linalg.inv(Tj).T)
		phi_hat[:,m*j:m*(j+1)] = np.matmul(Kj*rhsStages,Tj.T)
	for freqInd in range(HalfL,L):
		for stageInd in range(m):
			phi_hat[:,freqInd*m+stageInd] = np.conj(phi_hat[:,m*(L-freqInd)+stageInd])
	for stageInd in range(m):
		phi_hat[:,stageInd:m*L:m] = op.rescale_ifft(phi_hat[:,stageInd:m*L:m],N,T)
	phi_sol = phi_hat[:,:m*N]
	if first_value_is_t0:
		phi_sol = np.concatenate((np.zeros((len(phi_sol[:,0]),1)),phi_sol),axis=1)
	if flag_1Dinput:
		phi_sol = phi_sol[0,:]
	info = dict(Phi=g*np.pi,lls=lls,Kerr=np.abs(K_int-K_ex))
	return phi_sol,count_evals,info
