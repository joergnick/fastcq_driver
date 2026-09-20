import numpy as np
import bempp.api
from td_solvers import scattering_solution,spherical_Incident_wave
import warnings


#warnings.filterwarnings("error",category=ComplexWarning)
N = 100
T = 5
m = 1
dx = 1

n_grid_points=10
x_a=-1.5
x_b=1.5
y_a=-1.5
y_b=3.5
plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
Points = np.vstack( ( plot_grid[0].ravel() , plot_grid[1].ravel() , 0.0*np.ones(plot_grid[0].size) ) )
#Points
#Points = np.array([[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.1]])
import time
#sol = scattering_solution(N,T,m,dx,Points)
#print(str((end-start)/60)+" time for scattering problem.")
sol = np.load('data/sol.npy')
u_inc=spherical_Incident_wave(C=-10,t_0=0,center = [3,0,0],outgoing=True)
def u_inc(tj,x):
    return np.exp(-10*(-tj-x[0]+4)**2)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(sol.shape)
print(Points.shape)
r = np.linalg.norm(Points,axis=0)
print(r.shape)
for j in range(N):
    tj = T*j*1.0/N
    inc = np.array([u_inc(tj,p) for p in Points.T])
    tot = np.real(inc + sol[:,j])
    tot[r<1]=0
    plt.figure()
    plt.imshow(tot.reshape(n_grid_points,n_grid_points))
    plt.colorbar()
    plt.clim([-1,1])
    plt.savefig('data/wave_images/img'+str(j)+'.png')
    plt.close()

