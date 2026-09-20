import numpy as np
import bempp.api
print(bempp.api.__version__)
import sys
from td_solvers import scattering_solution,spherical_Incident_wave
### GENERATED DATA WITH:

n_grid_points = 10
x_a=-1.5
x_b=3.5
y_a=-1.5
y_b=3.5
plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
Points = np.vstack( ( plot_grid[0].ravel() , plot_grid[1].ravel() , 0.0*np.ones(plot_grid[0].size) ) )
## Cube
delta = 0.1
x1in = (-delta < Points[0]) & (Points[0] < 1+delta )
x2in = (-delta < Points[1]) & (Points[1] < 1+delta )
x3in = (-delta < Points[2]) & (Points[2] < 1+delta )
inside = x1in & x2in & x3in
am_outer = sum(1 - inside)

## Sphere
#r = np.linalg.norm(Points,axis=0)
#r_cutoff = 1.05
#inside = r< r_cutoff
#am_outer = sum(r>= r_cutoff)
#Points = np.array([[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.1]])
import time

T = 8
dxref = 0.05
am_time = 16
Nref  = 5*2**(am_time+2)

solref,counter = scattering_solution(Nref,T,dxref,Points,fast=False,lms=1)

#print("Compression rate for reference solution: "+str(counter/(Nref)))
#counterref = counter
np.save('data/solref.npy',solref)
#solref = np.load('data/solref.npy')
#print("Computed reference solution, shape "+str(solref.shape)+".")
solref[inside] = 0
#solref[r<r_cutoff] = 0
Ns =np.zeros((am_time,))
counters =np.zeros((am_time,))
## Right now only am_space = 1!!
am_space = 1
err = np.zeros((am_space,am_time))
errf = np.zeros((am_space,am_time))
errfvsn = np.zeros((am_space,am_time))

durations = np.zeros((am_space,am_time))
durationsf = np.zeros((am_space,am_time))
import scipy.io
res = dict()
for i_sp in range(am_space):
    dx = dxref
    #dx = 2**(-i_sp*1.0/2)
    for i_t in range(am_time):
        N= 10*2**(i_t)
        Ns[i_t] =N
        start = time.time()
        solf,counter = scattering_solution(N,T,dx,Points,fast=True)
        end = time.time()
        counters[i_t] = counter
        durationsf[i_sp,i_t] = (end-start)*1.0/60
        solf[inside] = 0
        start = time.time()
        sol,counter = scattering_solution(N,T,dx,Points,fast=False)
        end = time.time()
        durations[i_sp,i_t] = (end-start)*1.0/60
        sol[inside] = 0

        speed = Nref/N
        resc_ref = np.zeros((len(Points[0]),N+1))
        for j in range(N+1):
            resc_ref[:,j] = solref[:,int(j*speed)]

        err[i_sp,i_t] = 1.0/np.sqrt(am_outer*N/T)*np.linalg.norm(sol-resc_ref)
        errf[i_sp,i_t] = 1.0/np.sqrt(am_outer*N/T)*np.linalg.norm(solf-resc_ref)
        errfvsn[i_sp,i_t] = 1.0/np.sqrt(am_outer*N/T)*np.linalg.norm(solf-sol)       

        #err[i_sp,i_t] = 1.0/np.sqrt(am_outer)*np.linalg.norm(sol[:,-1]-solref[:,-1])
        #errf[i_sp,i_t] = 1.0/np.sqrt(am_outer)*np.linalg.norm(solf[:,-1]-solref[:,-1])
        #errfvsn[i_sp,i_t] = 1.0/np.sqrt(am_outer)*np.linalg.norm(solf[:,-1]-sol[:,-1])
        print("Ns: ",Ns)
        print("Error: ")
        print(err)
        print("Errorfast: ")
        print(errf)
        print("Errorfastvsnormal: ")
        print(errfvsn)
        print("Compression rate")
        print(counters[:i_t+1]/(Ns[:i_t+1]))
        end = time.time()
        print("Duration: ",end-start)
        res["counterref"] = 1
        res["counters"]   = counters
        res["durations"] = durations
        res["durationsf"] = durationsf
        res["Ns"]      = Ns
        res["err"]     = err
        res["errf"]    = errf
        res["errfvsn"] = errfvsn
        scipy.io.savemat("data/errorsh005cube.mat",res)
