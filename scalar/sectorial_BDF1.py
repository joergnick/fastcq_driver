import numpy as np
from cq_helpers import evalCQ,evalfastCQ#Import library of functions for CQ based on linear multistep methods
import matplotlib.pyplot as plt

# Data
g = lambda t : np.sin(t)**2
def gd(t):
    I = t < 2
    res = 2*np.cos(t)*np.sin(t)
    res[I] = 0
    return res

# Hyperbolic kernel
Ks = lambda s: s**1/(1-np.exp(-np.power( s , 2.0/3)))
T = 5
am = 13
Ns = np.zeros((am,))
Ls = np.zeros((am,))
errsCQ_vs_fastCQ = np.zeros((am,))
errsCQ_vs_ex = np.zeros((am,))
errsCQTR_vs_ex = np.zeros((am,))
errsfastCQ_vs_ex = np.zeros((am,))
N_ref=20*2**(am+5)
uref = evalCQ(g,Ks,N_ref,T,lms=0,prefactL=2)
import scipy.io
res = dict()
for j in range(am):
    N = 20*2**j
    Ns[j] = N
    dt =T/N
    import time 
    start = time.time()
    ufast,counterevals = evalfastCQ(g,Ks,N,T,lms=0,c=1,param=2,prefactL=1,domain="Sectorial",gamma=1,method="BDF1")
    Ls[j] = counterevals
    ucq = evalCQ(g,Ks,N,T,lms=0,prefactL=1)
    uTR = evalCQ(g,Ks,N,T,lms=1,prefactL=1)
    ts = np.linspace(0,T,N+1)
    ex = uref[::N_ref//N]
    errsCQ_vs_fastCQ[j] =np.max(np.abs(ucq-ufast))
    errsCQ_vs_ex[j] = np.sqrt(dt)*np.linalg.norm(np.abs(ucq-ex))
    errsCQTR_vs_ex[j] =np.max(np.abs(uTR-ex))
    errsfastCQ_vs_ex[j] =np.sqrt(dt)*np.linalg.norm(np.abs(ufast-ex))
    end = time.time()
    print("N: ",N," Time: ",end-start)
    res["counters"]   = Ls
    res["Ns"]      = Ns
    res["err"]     = errsCQ_vs_ex
    res["errTR"]   = errsCQTR_vs_ex
    res["errf"]    = errsfastCQ_vs_ex
    res["errfvsn"] = errsCQ_vs_fastCQ
    scipy.io.savemat("data/err_BDF1_sect.mat",res)
 
plt.rc('text', usetex=True)
plt.figure(figsize=(10, 6))
plt.subplot(1,2,2)
plt.loglog(Ns,errsCQ_vs_fastCQ,label=r"$||u_{\mathbf{cq}}-u_{\mathbf{fast}}||$",marker='s')
plt.loglog(Ns,Ns**(-1),label=r"$\mathcal{O}(N^{-1})$",linestyle='dashed')
plt.xlabel("Amount of timesteps (N)")
plt.ylabel("Maximal error")
plt.title("Error caused by interpolation")
plt.grid(True)
plt.legend()
plt.subplot(1,2,1)
plt.loglog(Ns,Ns**(-1),label=r"$\mathcal{O}(N^{-1})$",linestyle='dashed')
plt.loglog(Ns,errsCQ_vs_ex,label='Implicit Euler',marker='s')
plt.loglog(Ns,errsCQTR_vs_ex,label='Trapezoidal rule',marker='s')
plt.loglog(Ls,errsfastCQ_vs_ex,label='Modified Implicit Euler',marker='o')
plt.xlabel("Amount of Laplace evaluations (L)")
plt.ylabel("Error of the full scheme")
plt.grid(True)
plt.legend()
plt.title("Convergence plots for several methods")
plt.savefig("err_BDF1_sect.pdf",bbox_inches='tight')
#plt.show()


