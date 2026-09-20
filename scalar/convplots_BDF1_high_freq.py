import numpy as np
from cq_helpers import evalCQ,evalfastCQ#Import library of functions for CQ based on linear multistep methods
import matplotlib.pyplot as plt
# Data
#g = lambda t : 1.0/2*(t - np.sin(t)*np.cos(t))
varepsilon = 10**(-1)
g = lambda t : np.sin(t/varepsilon)**2
def gd(t):
    I = t < 0
    res = 2/varepsilon*np.cos(t/varepsilon)*np.sin(t/varepsilon)
    res[I] = 0
    return res


# Hyperbolic kernel
Ks = lambda s: s/(1-np.exp(-s))
T = 5
am = 13
Ns = np.zeros((am,))
Ls = np.zeros((am,))
errsCQ_vs_fastCQ = np.zeros((am,))
errsCQ_vs_ex = np.zeros((am,))
errsCQTR_vs_ex = np.zeros((am,))
errsfastCQ_vs_ex = np.zeros((am,))


import scipy.io
import time 
res = dict()
for j in range(am):
    N = 20*2**j
    Ns[j] = N
    dt =T/N
    start =time.time()
    ufast,counterevals = evalfastCQ(g,Ks,N,T,lms=0,c=2,param=1,prefactL=1)
    Ls[j] = counterevals
    ucq = evalCQ(g,Ks,N,T,lms=0,prefactL=1)
    uTR = evalCQ(g,Ks,N,T,lms=2,prefactL=1)
    ts = np.linspace(0,T,N+1)
    ex = gd(ts)+gd(ts-1)+gd(ts-2)+gd(ts-3)+gd(ts-4)
    errsCQ_vs_fastCQ[j] =np.max(np.abs(ucq-ufast))
    errsCQ_vs_ex[j] =np.max(np.abs(ucq-ex))
    errsCQTR_vs_ex[j] =np.max(np.abs(uTR-ex))
    errsfastCQ_vs_ex[j] =np.max(np.abs(ufast-ex))
    end = time.time()
    res["counters"]   = Ls
    res["Ns"]      = Ns
    res["err"]     = errsCQ_vs_ex
    res["errTR"]     = errsCQTR_vs_ex
    res["errf"]    = errsfastCQ_vs_ex
    res["errfvsn"] = errsCQ_vs_fastCQ
    scipy.io.savemat("data/err_BDF1_HS_high_freq.mat",res)
    print("N: ",N," Time: ",end-start)
    
plt.figure(figsize=(10, 6))
plt.rc('text', usetex=True)
plt.subplot(1,2,2)
plt.loglog(Ns,errsCQ_vs_fastCQ,label=r"$||u_{\mathbf{cq}}-u_{\mathbf{fast}}||$",marker='s')
plt.loglog(Ns,Ns**(-0.5),linestyle='dashed',label=r"$O(L^{-1/2})$")
plt.xlabel("Amount of timesteps $N$")
plt.ylabel("Maximal error")
plt.title("Error caused by interpolation")
plt.grid(True)
plt.legend()
plt.subplot(1,2,1)
plt.loglog(Ns,errsCQ_vs_ex,label='Implicit Euler',marker='s')
plt.loglog(Ls,errsfastCQ_vs_ex,label='Modified Implicit Euler',marker='o')
plt.loglog(Ns[2:],5*Ns[2:]**(-0.5),linestyle='dashed',label=r"$O(L^{-1/2})$")
plt.loglog(Ns[2:-3],60*Ns[2:-3]**(-1),linestyle='dashed',label=r"$O(L^{-1})$")
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel("Error of the full scheme")
plt.grid(True)
plt.legend()
plt.title("Convergence plot for several methods")
plt.savefig('err_BDF1_HS_high_freq.pdf')

