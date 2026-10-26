import numpy as np
import scipy.io
import time
from cq_helpers import evalCQ,evalfastCQ

## Data of the multistep methods (implicit Euler, BDF-2, trapezoidal rule) for the comparison plot
## scalarRK/paperplot_comparison.py, with the same problem and parameters as convplots_BDF1.py and
## convplots_BDF2.py. Laplace evaluations are counted on one side of the real axis only: the
## frequencies of classic CQ and the interval borders of the parsimonious method are symmetric
## about 1/2, so the other half follows by conjugation (L = N/2 for classic CQ).
## Both the maximal and the weighted l^2 error are saved.
g = lambda t : np.sin(t)**2
def gd(t):
    I = t < 0
    res = 2*np.cos(t)*np.sin(t)
    res[I] = 0
    return res
Ks = lambda s: s/(1-np.exp(-s))
T = 5
am = 13
keys = ["N","L_classic","L_BDF1f","L_BDF2f","err_BDF1_max","errf_BDF1_max","err_BDF2_max","errf_BDF2_max","err_TR_max",
        "err_BDF1_l2","errf_BDF1_l2","err_BDF2_l2","errf_BDF2_l2","err_TR_l2"]
res = {k:np.zeros(am) for k in keys}
for j in range(am):
    N = 20*2**j
    dt = T/N
    start = time.time()
    ts = np.linspace(0,T,N+1)
    ex = gd(ts)+gd(ts-1)+gd(ts-2)+gd(ts-3)+gd(ts-4)
    u = dict()
    u["BDF1"] = evalCQ(g,Ks,N,T,lms=0,prefactL=1)
    u["BDF1f"],count1 = evalfastCQ(g,Ks,N,T,lms=0,c=2,param=1,prefactL=1)
    u["BDF2"] = evalCQ(g,Ks,N,T,lms=1,prefactL=1)
    u["BDF2f"],count2 = evalfastCQ(g,Ks,N,T,param=1,lms=1,c=1,prefactL=1,method="BDF2")
    u["TR"] = evalCQ(g,Ks,N,T,lms=2,prefactL=1)
    res["N"][j] = N
    res["L_classic"][j] = N/2.0
    res["L_BDF1f"][j] = count1/2.0
    res["L_BDF2f"][j] = count2/2.0
    for name,key in [("BDF1","err_BDF1"),("BDF1f","errf_BDF1"),("BDF2","err_BDF2"),("BDF2f","errf_BDF2"),("TR","err_TR")]:
        res[key+"_max"][j] = np.max(np.abs(u[name]-ex))
        res[key+"_l2"][j]  = np.sqrt(dt)*np.linalg.norm(u[name]-ex)
    print("N: ",N," L classic/BDF1f/BDF2f: ",N/2.0,count1/2.0,count2/2.0," Time: ",time.time()-start)
scipy.io.savemat("data/paper_LMS_HS.mat",res)
