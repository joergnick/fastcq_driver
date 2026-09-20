import numpy as np
import bempp.api
import sys
sys.path.append('cq')
from cq_helpers import evalCQvectorial,evalfastCQvectorial
from th_solvers import th_scat
#from inverse_helpers import precomp
class spherical_Incident_wave:
    def __init__(self, C=-5,t_0=3,center = [0,0,0],outgoing=False):
        self.C = C
        self.t_0 = t_0
        self.center = center
        if outgoing:
            self.t_factor = -1 
        else:
            self.t_factor = 1
    def eval(self,t,x):
        #x_trans=x-self.center
        x_trans   = [0,0,0]
        x_trans[0]=x[0]-self.center[0]
        x_trans[1]=x[1]-self.center[1]
        x_trans[2]=x[2]-self.center[2]
        t_0 = self.t_0
        t   = self.t_factor*t
        C   = self.C
        y=np.exp(C*(np.linalg.norm(x_trans)-t_0+t)**2)/np.linalg.norm(x_trans)
        return y
#
#def create_rhs(N,T,grid):
#    dp0_space = bempp.api.function_space(grid,"P",1)
#    p1_space  = bempp.api.function_space(grid,"P",1)
#    m = 1
#    time_points = np.linspace(0,T,N+1)
#    #time_points = rk.get_time_points(T)
#    dof0=dp0_space.global_dof_count
#    dof1=p1_space.global_dof_count
#    dof=dof0+dof1
#    rhs=np.zeros((dof,m*N+1))
#    u_inc=spherical_Incident_wave(C=-10,t_0=0,center = [3,0,0],outgoing=True)
#    #u_inc=spherical_Incident_wave(C=-10,t_0=3,center = [2,0,0],outgoing=True)
#    ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
#    u_incs=np.zeros((dof1,m*N+1))   
#    pnu_incs=np.zeros((dof0,m*N+1))
#    for j in range(0,m*N+1):
#        #tj=j*T*1.0/N
#        tj = time_points[j]
#         
#        def u_inc_fun(x,normal,domain_index,result):
#            result[0]=np.exp(-10*(-tj-x[0]+4)**2)
#            #result[0]=u_inc.eval(tj,x)
#        gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
#
#        u_incs[:,j]=gridfun_inc.coefficients
#        pnu_incs[:,j]=gridfun_neu.coefficients
#
#        rhs[0:dof1,j]=gridfun_inc.coefficients
#        rhs[dof1:dof0+dof1,j]=gridfun_neu.coefficients
#    return rhs
import psutil
def create_rhs(N,T,grid):
    dp0_space = bempp.api.function_space(grid,"P",1)
    p1_space  = bempp.api.function_space(grid,"P",1)
    m = 1
    time_points = np.linspace(0,T,N+1)
    #time_points = rk.get_time_points(T)
    dof0=dp0_space.global_dof_count
    dof1=p1_space.global_dof_count
    print("GLOBAL DOF COUNT:",str(dof0))
    dof=dof0
    rhs=np.zeros((dof,m*N+1))
    #u_inc=spherical_Incident_wave(C=-10,t_0=0,center = [2,0,0],outgoing=True)
    #u_inc=spherical_Incident_wave(C=-10,t_0=3,center = [2,0,0],outgoing=True)
    ident_1=bempp.api.operators.boundary.sparse.identity(p1_space,p1_space,p1_space)
    u_incs=np.zeros((dof1,m*N+1))   
    pnu_incs=np.zeros((dof0,m*N+1))
    for j in range(0,m*N+1):
        #tj=j*T*1.0/N
        tj = time_points[j]
        def uinc(tj,x):
            uival = 0*x[0]
            #uival[np.abs((-tj-x[0]+4))<0.25]=1
            #uival=np.exp(-10*(-tj-x[0]+4)**2)
            uival=np.exp(-10*(x[0]-tj+4)**2)
            #if np.abs((-tj-x[0]+4))<0.25:
            #    uval = 1
            #else:
            #    uval = 0
            return uival
        #def uinc(tj,x):
        #    if np.abs((-tj-x[0]+4))<0.25:
        #        uval = 1
        #    else:
        #        uval = 0
        #    return uval
        #@bempp.api.real_callable
        #def u_inc_fun(x,normal,domain_index,result):
        #    result[0] = uinc(tj,x) 
        #    #result[0]=np.exp(-10*(-tj-x[0]+4)**2)
        #    #result[0]=np.exp(-10*(-tj-x[0]+4)**2)
        #gridfun_inc=bempp.api.GridFunction(p1_space,fun=u_inc_fun)
        rhs[:,j]=uinc(tj,grid.vertices)
        
        percent = psutil.virtual_memory()[2]
        if percent>95:
            print("On right-hand side, memory use: ",percent)
        #rhs[:,j]=gridfun_inc.coefficients
    #import matplotlib.pyplot as plt
    #plt.semilogy(np.linalg.norm(rhs,axis = 0))
    #plt.show()
    return rhs

def forward_solve(N,T,m,grid,Points):
    rhs = create_rhs(N,T,grid,m)
    def th_operator(s,b):
        return th_scat(s,b,grid,Points)
    sol = evalfastCQvectorial(rhs, th_operator, N, T, lms = 0,c=1,param=1,prefactL=2,gamma=None,method="BDF1",domain="Half-space")
    #sol = td_operator.apply_RKconvol(rhs,T,show_progress=False,method="RadauIIA-"+str(m))
    return np.real(sol)

def scattering_solution(N,T,dx,Points,grid = None,fast = True,lms = 0,tol=None):
    grid = bempp.api.shapes.cube(h=dx)
    #grid = bempp.api.shapes.sphere(h=dx)
    bempp.api.GLOBAL_PARAMETERS.quadrature.regular=6
    bempp.api.GLOBAL_PARAMETERS.quadrature.singular=6
#
#    nodes = grid.leaf_view.vertices
#    elements = grid.leaf_view.elements
#    nodes = 0.5*nodes
#    shift = 0*nodes
#    shift[1,:] = 0.5
#    nodes = nodes + shift
#    grid = bempp.api.grid_from_element_data(nodes,elements)
    rhs = create_rhs(N,T,grid)
    def th_operator(s,b):
        return th_scat(s,b,grid,Points)
    if fast:
        sol,counter,g_hat = evalfastCQvectorial(rhs, th_operator, N, T, lms = 0,c=1,param=1,prefactL=1,gamma=None,method="BDF1",domain="Half-space")
        #sol,counter,g_hat = evalfastCQvectorial(rhs, th_operator, N, T, lms = 0,c=np.power(N,0.2),param=1,prefactL=2,gamma=None,method="BDF1",domain="Half-space")
    else:
        sol,g_hat = evalCQvectorial(rhs, th_operator, N, T, lms = lms,prefactL=1,tol = tol)
        counter = N
    #sol2,counter,g_hat2 = fastcopy(rhs, th_operator, N, T, lms = 0,c=1,param=1,prefactL=2,gamma=None,method="BDF1",domain="Half-space")
    #sol2,counter,g_hat2 = evalfastCQvectorial(rhs, th_operator, N, T, lms = 0,c=np.power(N,0.2),param=0.8,prefactL=2,gamma=None,method="BDF1",domain="Half-space")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.semilogy(np.linspace(0,T,N+1),np.linalg.norm(sol,axis=0)) 
    #plt.show()
    #plt.semilogy(np.linalg.norm(g_hat.real-g_hat2.real),linestyle='dashed') 
    plt.savefig('testimage.png')
    ##print("Error:",np.linalg.norm(sol-sol2))
    ##print("EVALUATIONS NEEDED:", counter)
    #td_operator = Conv_Operator(th_operator)
    #sol = td_operator.apply_RKconvol(rhs,T,show_progress=False,method="RadauIIA-"+str(m))
    #return np.real(sol[:,0::m])
    return sol,counter
