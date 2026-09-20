import bempp.api
import scipy

grid = bempp.api.shapes.cube(h=1)
grid.plot()

def th_slp_solve(s,bs,fspace):
    Vs = bempp.api.operators.boundary.modified_helmholtz.single_layer(fspace,fspace,fspace,s)
    Mm1Vs = Vs.strong_form()
    sol = 0*bs
    for b_ind in range(len(bs[0,:])):
        sol[:,b_ind] = scipy.sparse.linalg.gmres(Mm1Vs,bs[b_ind],tol=10**(-6))
    return sol

def rhs(t,fspace):
        return 0    
