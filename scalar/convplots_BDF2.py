import numpy as np
from cq_helpers import evalCQ,evalfastCQ#Import library of functions for CQ based on linear multistep methods
import matplotlib.pyplot as plt

# Data
g = lambda t : np.sin(t)**2
def gd(t):
    I = t < 0
    res = 2*np.cos(t)*np.sin(t)
    res[I] = 0
    return res

# Hyperbolic kernel
#Ks = lambda s: np.exp(-s)
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
res = dict()
for j in range(am):
    N = 20*2**j
    Ns[j] = N
    dt =T/N
    # cq_lms
    import time 
    start =time.time()
    #ufast,counterevals = evalTchebyCQ(g,Ks,N,T,param=0.01,lms=1,c=np.sqrt(dt),method="BDF2",prefactL=2)
    #ufast,counterevals = evalTchebyCQ(g,Ks,N,T,param=np.log(N)*0.5,lms=1,c=2,method="BDF2")
    ufast,counterevals = evalfastCQ(g,Ks,N,T,param=1,lms=1,c=1,prefactL=1,method="BDF2")
    #ufast,counterevals = evalfastCQ(g,Ks,N,T,param=0.01,lms=1,c=np.sqrt(dt),method="BDF2")
    #ufast,counterevals = evalTchebyCQ(g,Ks,N,T,param=1,lms=1,c=1,method="BDF2")
    Ls[j] = counterevals
    ucq = evalCQ(g,Ks,N,T,lms=1,prefactL=1)
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
    scipy.io.savemat("data/err_BDF2_HS.mat",res)
    print("N: ",N," Time: ",end-start)
    

plt.figure(figsize=(10, 6))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.subplot(1,2,2)
plt.loglog(Ns,errsCQ_vs_fastCQ,label=r"$||u_{\mathbf{cq}}-u_{\mathbf{fast}}||$",marker='s')
plt.loglog(Ns,Ns**(-0.5),linestyle='dashed',label=r"$O(L^{-1/2})$")
plt.loglog(Ns,10*Ns**(-1),linestyle='dashed',label=r"$O(L^{-1})$")
plt.title("Error caused by interpolation")
plt.xlabel("Amount of timesteps $N$")
plt.ylabel("Maximal error")
plt.grid(True)
plt.legend()
#plt.show()
plt.subplot(1,2,1)
plt.loglog(Ns,errsCQ_vs_ex,label='BDF-2',marker='s')
plt.loglog(Ls,errsfastCQ_vs_ex,label='Modified BDF-2',marker='o')
plt.loglog(Ns,errsCQTR_vs_ex,label='TR',marker='d')
plt.loglog(Ns[2:],3*Ns[2:]**(-0.66),linestyle='dashed',label=r"$O(L^{-2/3})$")
#plt.loglog(Ns[2:],5*Ns[2:]**(-0.5),linestyle='dashed',label=r"$O(L^{-1/2})$")
plt.loglog(Ns[2:-4],100*Ns[2:-4]**(-4.0/3),linestyle='dashed',label=r"$O(L^{-4/3})$")
#plt.loglog(Ns[2:-3],60*Ns[2:-3]**(-1),linestyle='dashed',label=r"$O(L^{-1})$")
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel("Error of the full scheme")
plt.title("Convergence plot for several methods")
plt.grid(True)
plt.legend()
plt.savefig('err_BDF2_HS.pdf')
#plt.show()


