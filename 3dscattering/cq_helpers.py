import numpy as np
from scipy.fft import fft, ifft
from interval_helpers import create_intervals
from tcheby_helpers import piecewise_interpolation, lagrange_basis,tchebycheff_nodes,tcheb_interpolation,lagrange_interpolation

BDF1 = 0
BDF2 = 1
TR   = 2
TTR  = 3

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

def make_EPS(method,dt,cvgz_rate,p):
    mu = 1
    k = 0.5
    if method=="BDF1":
        EPS = 0.001*dt**(2*k+2*mu)
        #EPS = 10**(-10)
        #EPS = dt**(-mu)*cvgz_rate**(-p)
        #EPS = dt**(4)
        #EPS = dt**(4)
        #EPS = 10**(-14)

    elif method=="BDF2":
        #EPS = dt**6
        EPS = dt**(8/3+2)
        #EPS = dt**(8/3+2)

    else:
        EPS = 10**(-14)
    EPS = max(10**(-10),EPS)
    if EPS == 10**(-14):
        print("Minimum EPS reached. ")
    return EPS
import math
#def make_degrees(method,dt,cvgz_rate,borders,EPS):
#    if method == "BDF1":
#        #degrees = max(6,int(math.log(np.sqrt(EPS),cvgz_rate) ))*np.ones((len(borders)-1,))
#        #degrees = max(6,int(np.ceil(math.log(np.sqrt(EPS),cvgz_rate) )))*np.ones((len(borders)-1,))
#        degrees = max(6,int(np.ceil(math.log(dt**(-4),cvgz_rate) )))*np.ones((len(borders)-1,))
#    else:
#        degrees = max(6,3+int((math.log(dt**(-4),cvgz_rate) )))*np.ones((len(borders)-1,))
#        #degrees = max(6,3+int((math.log(dt**(-0),cvgz_rate)-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
#    return degrees
def make_p(method,dt,cvgz_rate):
    mu = 2
    k = 0.5
    if method == "BDF1":
        #degrees = max(6,int(math.log(np.sqrt(EPS),cvgz_rate) ))*np.ones((len(borders)-1,))
        #degrees = max(6,int(np.ceil(math.log(np.sqrt(EPS),cvgz_rate) )))*np.ones((len(borders)-1,))
        tol = max(dt**(mu+2*k),10**(-10))
        #tol = max(dt**(mu+2*k-1),10**(-14))
        degrees = max(6,1+int(np.ceil( - math.log(tol,cvgz_rate) )))
    else:
        k = 4/3
        mu = 1
        tol = max(dt**(mu+2*k),10**(-11))
        degrees = max(6,1+int((-math.log(tol,cvgz_rate) )))
        print("p: " ,degrees)
        #degrees = max(6,3+int((math.log(dt**(-4),cvgz_rate) )))
        #degrees = max(6,3+int((math.log(dt**(-0),cvgz_rate)-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    return degrees

def evalfastCQvectorial(rhs, apply_th, N, T, lms = BDF1,c=1,param=1,prefactL=1,gamma=None,method="BDF1",domain="Half-space"):
    dt = T/N
    cvgz_rate = c*2
    p = make_p(method,dt,cvgz_rate)
    print("p = ",p)
    #EPS = dt**4
    EPS = make_EPS(method,dt,cvgz_rate,p)
    print("EPS: ",EPS)
    lam = EPS**(1.0/(2*N+1))
    #lam = 10**(-14/(2*N+1))
    prefactL = 1
    L=prefactL*N
    lams = lam**np.arange(L+1)
    ts = np.linspace(0,T,N+1)
    #gs = g(ts) 
    gs = rhs
    gvals =np.concatenate((gs,np.zeros((len(gs[:,0]),L-N))),axis=1)
    def deltatau(x):
        return dlt_lms(lam*np.exp(-2*np.pi*1j*x),lms)/dt
    evalpoints = np.arange(L+1)/(L+1)
    borders = create_intervals(c,lam,dt,N,method=method,domain=domain,gamma=gamma)
    counter_evals = 0
    i_js=len(borders)*[0]
    borderindex=1
    for ind_x in range(len(evalpoints)):
        if evalpoints[ind_x]>borders[borderindex]:
            i_js[borderindex]=ind_x
            borderindex += 1
    i_js[-1] = len(evalpoints)-1
    
    g_hat = fft(lams*gvals) 

    dummy = apply_th(deltatau(evalpoints[0])/dt,np.array([g_hat[:,0]]).T)
    u_hat = 1j*np.zeros((len(dummy),len(g_hat[0,:])))
    #u_hat = 0*g_hat
    import math
    #cvgz_rate = c+math.sqrt(1+c**2)
    #degrees = 1+max(6,int(1*param*(math.log(N,cvgz_rate)*(2+1+1)-math.log(10**(-8),cvgz_rate)) ))*np.ones((len(borders)-1,))
    #print('degree used: '+str(degrees[0]))
    import matplotlib.pyplot as plt
    #plt.figure()
    for j in range(1,len(borders)):
        a = borders[j-1]
        b = borders[j]
        d = p #degrees[j-1]
        if (i_js[j]-i_js[j-1]<=d+2) or (j==1) or (j==len(borders)-1):
            ## No interpolation
        #if i_js[j]-i_js[j-1]<=d or (j==1) or (j==len(borders)-1):
            #Kvals_loc = f(evalpoints[i_js[j-1]:i_js[j]])
            for ind_j in range(i_js[j-1],i_js[j]):
                #plt.scatter(np.real(deltatau(evalpoints[ind_j])),np.imag(deltatau(evalpoints[ind_j])))
                u_hat[:,ind_j] = apply_th(deltatau(evalpoints[ind_j]),g_hat[:,ind_j:ind_j+1])[:,0]
            counter_evals += i_js[j]-i_js[j-1]
        else:
            ## Interpolation
            xkj = tchebycheff_nodes(a,b,d)
            #Kvals_nodes = f(xkj)
            L = lagrange_basis(xkj, evalpoints[i_js[j-1]:i_js[j]])
            for ind_d in range(int(d)):
                u_hat[:,i_js[j-1]:i_js[j]] += apply_th(deltatau(xkj[ind_d]), L[ind_d,:]*g_hat[:,i_js[j-1]:i_js[j]])
                #plt.scatter(np.real(deltatau(xkj[ind_d])),np.imag(deltatau(xkj[ind_d])))
            counter_evals += d-1
    #plt.savefig("freqs_N"+str(N)+".png")
    #zs = lam*np.exp(-2*np.pi*1j*np.arange(N+1)/(N+1))
    endind = len(g_hat[0,:])-1
    u_hat[:,endind] = apply_th(deltatau(evalpoints[endind]),g_hat[:,endind:endind+1])[:,0]

    #plt.scatter(np.real(deltatau(evalpoints[endind])),np.imag(deltatau(evalpoints[endind])))
    u = (1/lams)*ifft(u_hat)
    return u[:,:N+1], counter_evals,u_hat

def evalCQvectorial(g, apply_th, N, T, lms = BDF1,prefactL=1,tol=None):
    dt = T/N
    #EPS = make_EPS("BDF1",dt)
    #lam = EPS**(1.0/(2*N+1))
    #if tol is None:
    #    lam = 10**(-14/(2*N+1))
    #else:
    #    lam = tol**(1.0/(2*N+1))
    lam = 10**(-14/(2*N+1))
    L= prefactL*N
    lams = lam**np.arange(L+1)
    ts = np.linspace(0,T,N+1)
    #gs = g(ts) 
    gs = g
    gvals =np.concatenate((gs,np.zeros((len(gs[:,0]),L-N))),axis=1)
    zs = lam*np.exp(-2*np.pi*1j*np.arange(L+1)/(L+1))
    g_hat = fft(lams*gvals)
    u_hat = g_hat*0
    dummy = apply_th(dlt_lms(zs[0],lms)/dt,np.array([g_hat[:,0]]).T)
    u_hat = 1j*np.zeros((len(dummy[:,0]),len(g_hat[0,:])))
    u_hat[:,0] = dummy[:,0]
    for l in range(0,len(g_hat[0,:])):
        u_hat[:,l] = apply_th(dlt_lms(zs[l],lms)/dt,g_hat[:,l:l+1])[:,0]
    u = (1/lams)*ifft(u_hat)
    return u[:,:N+1],u_hat



## Non-vectorial
def evalCQ(g, Ks, N, T, lms = BDF1,prefactL=2):
    dt = T/N
    tol = max(0,10**(-8))
    #tol = max(dt**6,10**(-10))
    lam =tol**(1.0/(2*N+1))
    #lam = 10**(-10/(2*N+1))
    prefactL = 10
    L= int(prefactL*N)
    lams = lam**np.arange(L+1)
    ts = np.linspace(0,T,N+1)
    gvals =np.concatenate((g(ts),np.zeros((L-N,))),axis=0)
    zs = lam*np.exp(-2*np.pi*1j*np.arange(L+1)/(L+1))
    u = (1/lams)*ifft(Ks(dlt_lms(zs,lms)/dt)*fft(lams*gvals))
    
    return u[:N+1]


def evalTchebyCQ(g, Ks, N, T, lms = BDF1,c=1,param=1,prefactL=2,gamma=None,method="BDF1",domain="Half-space"):
    dt = T/N
    lam = 10**(-14/(2*N+1))
    L=prefactL*N
    lams = lam**np.arange(L+1)
    ts = np.linspace(0,T,N+1)
    gvals =np.concatenate((g(ts),np.zeros((L-N,))),axis=0)
    def f(x):
        return Ks(dlt_lms(lam*np.exp(-2*np.pi*1j*x),lms)/dt)
    evalpoints = np.arange(L+1)/(L+1)
    borders = create_intervals(c,lam,dt,N,method=method,domain=domain,gamma=gamma)
    
    #degrees = 20*np.ones((len(borders)-1,))
    import math
    cvg = (c+math.sqrt(1+(c)**2))
    if lms == BDF1:
        degrees = 2+max(10,int(param*c**(-1)*(math.log(N,cvg)*(2+1+2)-math.log(10**(-8),cvg)) ))*np.ones((len(borders)-1,))
    if lms == BDF2:
        degrees = 2+max(10,int(param*c**(-1)*(math.log(N,cvg)*(2+1+2)-math.log(10**(-8),cvg)) ))*np.ones((len(borders)-1,))
    #degrees = max(6,int(param*c**(-1)*(-math.log(10**(-16),cvg)) ))*np.ones((len(borders)-1,))
    Kvals,counterevals = piecewise_interpolation(borders,degrees,evalpoints,f)
    import matplotlib.pyplot as plt
#    plt.figure
#    print(len(evalpoints))
#    plt.plot(evalpoints[:100],Kvals[:100].real,label='Real part',linestyle='dashed')
#    plt.plot(evalpoints[:100],Kvals[:100].imag,label='Imaginary part',linestyle='dashed')
#    plt.legend()
#    plt.savefig('Ks.png')
    print("Error = ",np.linalg.norm(Kvals-f(evalpoints)))
    #zs = lam*np.exp(-2*np.pi*1j*np.arange(N+1)/(N+1))
    u = (1/lams)*ifft(Kvals*fft(lams*gvals))
    
    return u[:N+1], counterevals





def evalfastCQ(g, Ks, N, T, lms = BDF1,c=1,param=1,prefactL=1,gamma=None,method="BDF1",domain="Half-space"):
    dt = T/N
    #EPS = make_EPS(method,dt)
    cvgz_rate = 2
    p = make_p(method,dt,cvgz_rate)
    EPS = make_EPS(method,dt,cvgz_rate,p)
    lam = EPS**(1.0/(2*N+1))
    #lam = EPS**(1.0/(2*N+1))
    #lam = 10**(-16/(2*N+1))
    L=int(prefactL*N)
    lams = lam**np.arange(L+1)
    ts = np.linspace(0,T,N+1)
    gvals =np.concatenate((g(ts),np.zeros((L-N,))),axis=0)
    def f(x):
        return Ks(dlt_lms(lam*np.exp(-2*np.pi*1j*x),lms)/dt)
    evalpoints = np.arange(L+1)/(L+1)
    borders = create_intervals(c,lam,dt,N,method=method,domain=domain,gamma=gamma)
    print("Amount of intervals: " +str(len(borders)-1))
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
    #cvgz_rate = c+math.sqrt(1+c**2)
    #degrees = max(6,int((math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    #degrees = max(5,int((-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    #degrees = make_degrees(method,dt,cvgz_rate,borders,EPS)
    degrees = -2+p * np.ones((len(borders)-1,))

    #if method == "BDF1":
    #    degrees = max(6,int((math.log(dt**(-0),cvgz_rate)-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    #else:
    #    degrees = max(6,int((math.log(dt**(-0),cvgz_rate)-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    #degrees = max(6,int((math.log(N,cvgz_rate)*(2)-math.log(np.sqrt(EPS),cvgz_rate)) ))*np.ones((len(borders)-1,))
    print("Degree: ",degrees[0])
    #degrees = max(6,int(1*param*(math.log(N,cvgz_rate)*(2+1+2)-math.log(10**(-8),cvgz_rate)) ))*np.ones((len(borders)-1,))
    eps_degree = 2*int(-math.log(10**(-15),cvgz_rate)) 
    if eps_degree <= degrees[0] :
        print("Max degree reached: "+str(eps_degree))
        degrees = eps_degree*np.ones((len(borders)-1,))
    #degrees = max(6,int(1*param*(math.log(N,cvgz_rate)*(2+1+1)-math.log(10**(-8),cvgz_rate)) ))*np.ones((len(borders)-1,))
    #degrees = max(6,int(1*param*c**(-1)*(math.log(N,1+c)*(2+1+2)-math.log(10**(-8),1+c)) ))*np.ones((len(borders)-1,))
    #print("Degrees: ", degrees)
    use_interpolation = [degrees[j]<len(evals_in_borders[j]) for j in range(len(degrees))]
    tt = np.linspace(0,1000,1001)
    for j in range(1,len(borders)):
        a = borders[j-1]
        b = borders[j]
       # a = evalpoints[i_js[j-1]]
       # b = evalpoints[i_js[j]]
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
            #counter_evals += d
    u_hat[-1] = f(evalpoints[-1])*g_hat[-1]

    u = (1/lams)*ifft(u_hat)
    
    return u[:N+1],counter_evals   #,g_hat

#
#def evalCQvectorialoverkill(g, apply_th, N, T, lms = BDF1,c=1,param=1,prefactL=2,gamma=None,method="BDF1",domain="Half-space"):
#    dt = T/N
#    lam = 10**(-14/(2*N+1))
#    L=prefactL*N
#    lams = lam**np.arange(L+1)
#    ts = np.linspace(0,T,N+1)
#    gs = g(ts) 
#    gvals =np.concatenate((gs,np.zeros((len(gs[:,0]),L-N))),axis=1)
#    def deltatau(x):
#        return dlt_lms(lam*np.exp(-2*np.pi*1j*x),lms)/dt
#    evalpoints = np.arange(L+1)/(L+1)
#    borders = create_intervals(c,lam,dt,N,method=method,domain=domain,gamma=gamma)
#    evals_in_borders = [] # Orders the 
#    ind_in_borders = [] # Orders the 
#    counter_evals = 0
#    i_js=len(borders)*[0]
#    borderindex=1
#    for ind_x in range(len(evalpoints)):
#        if evalpoints[ind_x]>borders[borderindex]:
#            i_js[borderindex]=ind_x
#            borderindex += 1
#    i_js[-1] = len(evalpoints)-1
#    
#    for j in range(1,len(borders)):
#        evals_in_borders.append([x for x in evalpoints if ((borders[j-1] <= x) and (x <= borders[j]))])
#        ind_in_borders.append([i for i in range(len(evalpoints)) if ((borders[j-1] <= evalpoints[i]) and (evalpoints[i] <= borders[j]))])
#    g_hat = fft(lams*gvals) 
#    u_hat = 0*g_hat
#    degrees = max(6,int(1*param*c**(-1)*(np.log(N)*(2+1+2)-np.log(10**(-8))) ))*np.ones((len(borders)-1,))
#    use_interpolation = [degrees[j]<len(evals_in_borders[j]) for j in range(len(degrees))]
#    tt = np.linspace(0,1000,1001)
#    for j in range(1,len(borders)):
#        a = borders[j-1]
#        b = borders[j]
#        a = evalpoints[i_js[j-1]]
#        b = evalpoints[i_js[j]]
#        d = degrees[j-1]
#        if i_js[j]-i_js[j-1]<=d or (j==1) or (j==len(borders)-1):
#            #Kvals_loc = f(evalpoints[i_js[j-1]:i_js[j]])
#            for ind_j in range(i_js[j-1],i_js[j]):
#                u_hat[:,ind_j] = apply_th(deltatau(evalpoints[ind_j]),g_hat[:,ind_j:ind_j+1])[:,0]
#            counter_evals += i_js[j]-i_js[j-1]
#        else:
#            xkj = tchebycheff_nodes(a,b,d)
#            #Kvals_nodes = f(xkj)
#            L = lagrange_basis(xkj, evalpoints[i_js[j-1]:i_js[j]])
#            for ind_d in range(int(d)):
#                u_hat[:,i_js[j-1]:i_js[j]] += apply_th(deltatau(evalpoints[ind_j]), L[ind_d,:]*g_hat[:,i_js[j-1]:i_js[j]])
#            counter_evals += d
#    #zs = lam*np.exp(-2*np.pi*1j*np.arange(N+1)/(N+1))
#    u_hat[:,-1] = apply_th(deltatau(evalpoints[-1]),np.ones((len(g_hat[:,0]),1))*g_hat[:,-1])[:,0]
#    u = (1/lams)*ifft(u_hat)
#    
#    return u[:,:N+1], counter_evals
#
#
