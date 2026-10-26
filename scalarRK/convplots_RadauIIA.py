import numpy as np
import scipy.io
import matplotlib.pyplot as plt

## Loading the results of convplots_RadauIIA2.py and convplots_RadauIIA3.py
data = dict()
for m in [2,3]:
	data[m] = scipy.io.loadmat("../scalar/data/err_RadauIIA"+str(m)+"_HS.mat")

plt.close('all')
plt.figure(figsize=(8, 6))
plt.rc('text', usetex=True)
markers = {2: ('s','o'), 3: ('d','^')}
for m in [2,3]:
	Ns       = data[m]["Ns"].ravel()
	counters = data[m]["counters"].ravel()
	err      = data[m]["err"].ravel()
	errf     = data[m]["errf"].ravel()
	plt.loglog(Ns,err,label="Radau IIA-"+str(m),marker=markers[m][0])
	plt.loglog(counters,errf,label="Modified Radau IIA-"+str(m),marker=markers[m][1])

## Reference slopes, anchored at the last classic data point
for m,p,lab,ls in [(2,0.75,r"$O(L^{-3/4})$",'dashed'),(3,1.25,r"$O(L^{-5/4})$",'dotted')]:
	Ns  = data[m]["Ns"].ravel()
	err = data[m]["err"].ravel()
	plt.loglog(Ns,3*err[-1]*(Ns/Ns[-1])**(-p),linestyle=ls,color='gray',label=lab)

plt.xlabel("Amount of Laplace evaluations $L$")
plt.ylabel("Error of the full scheme")
plt.title("Convergence plot for Radau IIA methods")
plt.grid(True)
plt.legend()
plt.savefig('errs_RadauIIA.pdf')
