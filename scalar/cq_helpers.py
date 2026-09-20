import numpy as np
from scipy.fft import fft, ifft
from interval_helpers import create_intervals
from tcheby_helpers import piecewise_interpolation, lagrange_basis,tchebycheff_nodes,tcheb_interpolation,lagrange_interpolation

BDF1 = 0
BDF2 = 1
TR = 2
TTR = 3

def dlt_lms(z,lms):
    if lms == BDF1:
        return 1-z
    elif lms == BDF2:
        return 1-z+(1/2)*(1-z)**2
    elif lms == TR:
        return 2*(1-z)/(1+z)
    elif lms == TTR:
        cs = [0.893817850529318,   0.684154908023834,   0.629642997466429]
        return 1-z+(1/2)*(1-z)**2+(1/4)*cs[0]*(1-z)**3+(1/8)*cs[1]*(1-z)**4+(1/16)*cs[2]*(1-z)**5
    else:
        return "No such LMS method"
import cmath

def make_EPS(method,dt,domain="Half-space"):
    if method=="BDF1":
        if domain=="Half-space":
            ## Half space
            EPS =0.01* dt**3
        else:
            ## Sectorial
            EPS = dt**4
    elif method=="BDF2":
        if domain=="Half-space":
            ## Half space
            EPS = 0.1*(dt)**(4/3+2)
        else:
            ## Sectorial
            EPS = 0.1*(dt)**(4/3+3)
    else:
        EPS = 10**(-14)
    EPS = max(10**(-16),EPS)
    if EPS == 10**(-14):
        print("Minimum EPS reached. ")
    return EPS
import math
def make_degrees(method,dt,cvgz_rate,borders,EPS):
    if method == "BDF1":
        degrees = max(6,int((math.log(dt**(-0),cvgz_rate)-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    else:
        degrees = max(6,3+int((math.log(dt**(-0),cvgz_rate)-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    return degrees

## Non-vectorial standard CQ, taken from https://github.com/lehelb/TDBIE-CQ-book
def evalCQ(g, Ks, N, T, lms = BDF1,prefactL=2):
    dt = T/N
    lam = 10**(-16/(2*N+1))
    L= int(prefactL*N)
    lams = lam**np.arange(L+1)
    ts = np.linspace(0,T,N+1)
    gvals =np.concatenate((g(ts),np.zeros((L-N,))),axis=0)
    zs = lam*np.exp(-2*np.pi*1j*np.arange(L+1)/(L+1))
    u = (1/lams)*ifft(Ks(dlt_lms(zs,lms)/dt)*fft(lams*gvals))
    
    return u[:N+1]



## Parsimonious adaption.
def evalfastCQ(g, Ks, N, T, lms = BDF1,c=1,param=1,prefactL=1,gamma=None,method="BDF1",domain="Half-space"):
    dt = T/N
    EPS = make_EPS(method,dt,domain = domain)
    lam = EPS**(1.0/(2*N+1))
    L=int(prefactL*N)
    lams = lam**np.arange(L+1)
    ts = np.linspace(0,T,N+1)
    gvals =np.concatenate((g(ts),np.zeros((L-N,))),axis=0)
    def f(x):
        return Ks(dlt_lms(lam*np.exp(-2*np.pi*1j*x),lms)/dt)
    evalpoints = np.arange(L+1)/(L+1)
    borders = create_intervals(c,lam,dt,N,method=method,domain=domain,gamma=gamma)
    evals_in_borders = [] # Orders the 
    ind_in_borders = [] # Orders the 
    counter_evals = 0
    i_js=len(borders)*[0]
    borderindex=1
    for ind_x in range(len(evalpoints)):
        if evalpoints[ind_x]>borders[borderindex]:
            i_js[borderindex]=ind_x
            borderindex += 1
    i_js[-1] = len(evalpoints)-1
    
    for j in range(1,len(borders)):
        evals_in_borders.append([x for x in evalpoints if ((borders[j-1] <= x) and (x <= borders[j]))])
        ind_in_borders.append([i for i in range(len(evalpoints)) if ((borders[j-1] <= evalpoints[i]) and (evalpoints[i] <= borders[j]))])
    g_hat = fft(lams*gvals) 
    u_hat = 0*g_hat
    import math
    cvgz_rate = c*2
    degrees = make_degrees(method,dt,cvgz_rate,borders,EPS)

    eps_degree = 2*int(-math.log(10**(-15),cvgz_rate)) 
    if eps_degree <= degrees[0] :
        print("Max degree reached: "+str(eps_degree))
        degrees = eps_degree*np.ones((len(borders)-1,))
    for j in range(1,len(borders)):
        a = borders[j-1]
        b = borders[j]
        d = degrees[j-1]
        if i_js[j]-i_js[j-1]<=d or (j==1) or (j==len(borders)-1):
            Kvals_loc = f(evalpoints[i_js[j-1]:i_js[j]])
            u_hat[i_js[j-1]:i_js[j]] = Kvals_loc*g_hat[i_js[j-1]:i_js[j]]
            counter_evals += i_js[j]-i_js[j-1]
        else:
            xkj = tchebycheff_nodes(a,b,d)
            Kvals_nodes = f(xkj)
            L = lagrange_basis(xkj, evalpoints[i_js[j-1]:i_js[j]])
            for ind_d in range(int(d)):
                u_hat[i_js[j-1]:i_js[j]] += Kvals_nodes[ind_d]*((L[ind_d,:]*g_hat[i_js[j-1]:i_js[j]]))
            counter_evals += d-1
    u_hat[-1] = f(evalpoints[-1])*g_hat[-1]

    u = (1/lams)*ifft(u_hat)
    
    return u[:N+1],counter_evals   