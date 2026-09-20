import numpy as np
from cq_helpers import evalCQ,evalfastCQvectorial,evalTchebyCQ#Import library of functions for CQ based on linear multistep methods
import matplotlib.pyplot as plt
# Data
#g = lambda t : 1.0/2*(t - np.sin(t)*np.cos(t))
g = lambda t : np.sin(t)**3#
Ks = lambda s: s**2 /(1-np.exp(-s))
def apply_th(s,b):
    return Ks(s)*b
#Ks = lambda s: s**2/(1-np.exp(-s))
T = 1
am = 12
Ns = np.zeros((am,))
Ls = np.zeros((am,))
errsCQ_vs_fastCQ = np.zeros((am,))
errsCQ_vs_ex = np.zeros((am,))
errsCQTR_vs_ex = np.zeros((am,))
errsfastCQ_vs_ex = np.zeros((am,))
N_ref =10*2**(am+1)
uref = evalCQ(g,Ks,N_ref,T,lms=1,prefactL=2)


tp = np.linspace(0,T,N_ref+1)
#plt.plot(g(tp))
#plt.show()
import scipy.io
res = dict()
for j in range(am):
    N = 10*2**j
    Ns[j] = N
    dt =T/N
    # cq_lms
    import time 
    start =time.time()
    #ufast,counterevals = evalTchebyCQ(g,Ks,N,T,lms=0,c=1,param=0.4,prefactL=2)
    tp = np.linspace(0,T,N+1)
    gs = np.zeros((1,N+1))
    gs[0,:] = g(tp)

    ufast,counterevals,uhat = evalfastCQvectorial(gs,apply_th,N,T,lms=0,c=1)
    #ufast,counterevals = evalfastCQvectorial(gs,Ks,N,T,lms=0,c=2,param=1,prefactL=1)
    #ufast,counterevals = evalfastCQ(g,Ks,N,T,lms=0,c=2,param=1,prefactL=1)
    
    #ufast,counterevals = evalfastCQ(g,Ks,N,T,lms=0,c=np.sqrt(dt),param=0.01,prefactL=2)

    ## In document:
    #ufast,counterevals = evalTchebyCQ(g,Ks,N,T,lms=0,c=np.power(N,0.1),param=1,prefactL=2)
    Ls[j] = counterevals
    ucq = evalCQ(g,Ks,N,T,lms=0,prefactL=1)
    uTR = evalCQ(g,Ks,N,T,lms=2,prefactL=1)
    ts = np.linspace(0,T,N+1)
    #ex = gd(ts)
    #ex = gd(ts)+gd(ts-1)+gd(ts-2)+gd(ts-3)+gd(ts-4)
    errsCQ_vs_fastCQ[j] =np.max(np.abs(ucq-ufast))
    #print(errsCQ_vs_fastCQ)
    ex = uref[::N_ref//N]
    errsCQ_vs_ex[j] =np.max(np.abs(ucq-ex))
    print(errsCQ_vs_ex)
    errsCQTR_vs_ex[j] =np.max(np.abs(uTR-ex))
    errsfastCQ_vs_ex[j] =np.max(np.abs(ufast-ex))
    end = time.time()
    print("N: ",N," Time: ",end-start)
    res["counters"]   = Ls
    #res["durations"] = durations
    #res["durationsf"] = durationsf
    res["Ns"]      = Ns
    res["err"]     = errsCQ_vs_ex
    #res["errTR"]     = errsCQTR_vs_ex
    res["errf"]    = errsfastCQ_vs_ex
    res["errfvsn"] = errsCQ_vs_fastCQ
    scipy.io.savemat("data/err_BDF1_HS.mat",res)

plt.figure(figsize=(10, 6))
plt.rc('text', usetex=True)
plt.subplot(1,2,2)
plt.loglog(Ns,errsCQ_vs_fastCQ,label=r"$||u_{\mathbf{cq}}-u_{\mathbf{fast}}||$",marker='s')
plt.loglog(Ns,Ns**(-0.5),linestyle='dashed',label=r"$O(L^{-1/2})$")
#plt.loglog(Ns,Ns**(-1),linestyle='dashed')
plt.xlabel("Amount of timesteps $N$")
plt.ylabel("Maximal error")
plt.title("Error caused by interpolation")
plt.grid(True)
plt.legend()
#plt.show()
plt.subplot(1,2,1)
plt.loglog(Ns,errsCQ_vs_ex,label='Implicit Euler',marker='s')
#plt.loglog(Ns,errsCQTR_vs_ex,label='Trapezoidal rule',marker='s')
plt.loglog(Ls,errsfastCQ_vs_ex,label='Modified Implicit Euler',marker='o')
plt.loglog(Ns[2:],5*Ns[2:]**(-0.5),linestyle='dashed',label=r"$O(L^{-1/2})$")
plt.loglog(Ns[2:-3],60*Ns[2:-3]**(-1),linestyle='dashed',label=r"$O(L^{-1})$")
plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel("Error of the full scheme")
plt.grid(True)
plt.legend()
plt.title("Convergence plot for several methods")
#plt.savefig("plots/BDF1_convergence.eps")
plt.show()


#
#plt.plot(ts,gd(ts)+gd(ts-1)+gd(ts-2)+gd(ts-3)+gd(ts-4),linestyle='dashed')
#plt.subplot(1,2,2)
##plt.semilogy(np.abs(ex.real-ucq.real))
#plt.plot(ex.real,linestyle='dashed')
#plt.show()
