import numpy as np
def create_intervals(c,rho,tau,N,method="BDF1",domain="Half-space",gamma=None):
    if domain== "Half-space":
        phik = 0
        if method=="BDF1":
            phik = np.sqrt(10*np.abs(np.log(rho))/(6*N))
        if method=="BDF2":
            phik = np.sqrt(10*np.abs(np.log(rho))/(6*N))
        else:
            phik = np.sqrt(tau)
        borders = [0,phik]
        while phik <0.1:
            if method=="BDF1":
                phik = phik + c**(-1)*8/3*np.abs(np.log(rho))   + c**(-1)*8/3*np.abs(np.log(np.cos(7/8*phik))) 

            if method=="BDF2":
                phik = phik + c**(-1)*np.abs(np.log(rho))   + 0.2*1.5*c**(-1)*phik**2
            borders.append(phik)
        if borders[-1]>0.5 :
            borders.pop()
        borders.append(0.5)


    if domain=="Sectorial":
        phik = 0
        phik = tau
        borders = [0,phik]
        while phik <0.3 :
            if method=="BDF1":
                phik = phik + c**(-1)*np.abs(np.log(rho)) +(c**(-1)*np.pi*gamma*phik)+0.5*c**(-1)*phik**2  #+c**(-1)*np.abs(np.log(np.cos(phik))) #+ 0.5*phik**2
            if method=="BDF2":
                phik = phik + c**(-1)*np.abs(np.log(rho))   + (c**(-1)*gamma*phik)+1.5*c**(-1)*phik**2
            borders.append(phik)
    borders[-1] = 0.5
    mirrored_borders = reversed([1-b for b in borders[:-1]])
    borders += mirrored_borders
    return borders
