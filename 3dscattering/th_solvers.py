
import bempp.api
import numpy as np
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
import time
import psutil
def ex_th_scat(s,b,grid,Points):
    u_ex  = np.exp(-s*np.linalg.norm(Points,axis=0))*1.0/(np.linalg.norm(Points,axis=0))
    u_ex[np.linalg.norm(Points,axis=0)<1.01] = 0
    return u_ex 
def th_scat(s,bs,grid,Points):
    #OrderQF = 9
    #tol= np.finfo(float).eps
    #bempp.api.global_parameters.hmat.eps=10**-7
    #bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
    #bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
    #bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
    #bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
    #bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
    #bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
    #bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
    #bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
    #bempp.api.global_parameters.quadrature.double_singular = OrderQF
    if np.linalg.norm(np.array(bs))<10**(-12):
        return np.zeros((len(Points[0]),len(bs[0,:])))
    #Define space
    dp0_space = bempp.api.function_space(grid,"P",1)
    #dp0_space = bempp.api.function_space(grid,"DP",0)
    p1_space = bempp.api.function_space(grid, "P" ,1)
    dof0 = dp0_space.global_dof_count
    dof1 = p1_space.global_dof_count
    ## Definition of Operators
    slp = bempp.api.operators.boundary.helmholtz.single_layer(dp0_space,dp0_space,dp0_space,1j*s)
    #slp = bempp.api.operators.boundary.modified_helmholtz.single_layer(dp0_space,dp0_space,dp0_space,s)
    ## Potential operators
    transformed_s=False
    if np.imag(s)==0:
        print("Transformation, s = ",s)
        s = s-10**(-14)*1j
        #s = s+10**(-12)*1j
        transformed_s = True

    percent = psutil.virtual_memory()[2]
    if percent>95:
        print("TH solve, memory use: ", percent)
    slp_pot = bempp.api.operators.potential.helmholtz.single_layer(dp0_space,Points,1j*s)
    #slp_pot = bempp.api.operators.potential.modified_helmholtz.single_layer(dp0_space,Points,s)
     
    from scipy.sparse.linalg import gmres
    slp_strong = slp
    #slp_strong = slp.strong_form()
    #phiscat = 1j*np.zeros((dof0,len(bs[0,:])))
    uscat = 1j*np.zeros((len(Points[0]),len(bs[0,:])))
    sysnum = len(bs[0,:])
    bgrids = [bempp.api.GridFunction(p1_space,coefficients=bs[:,j]) for j in range(sysnum) ] 
    #lufactors = bempp.api.linalg.compute_lu_factors(slp_strong)
    ## Catch transformed case?
    #if transformed_s:
    #    #slp_strong = slp.strong_form()
    #    #varph,info=gmres(slp_strong,bs[:,j],maxiter=300,tol=10**(-7))
    #    for j in range(len(bs[0,:])):
    #        #varph,info=gmres(slp_strong,bs[:,j],maxiter=300,tol=10**(-7))
    #        varph_gf = 0*bgrids[0]
    #        #varph_gf = bempp.api.linalg.lu(slp,bgrids[j],lu_factor=lufactors)
    #        #if info != 0:
    #        #    print("info: ", info)
    #        #varph_gf = bempp.api.GridFunction(p1_space,coefficients=varph)
    #        uscat[:,j]=slp_pot*varph_gf
    #    return uscat

    #print("Frequency: ",s)
    #print("Matrix: " , bempp.api.as_matrix(slp.weak_form()))
    slp_strong = slp.strong_form()
    lufactors = scipy.linalg.lu_factor(bempp.api.as_matrix(slp.weak_form()))
    for j in range(len(bs[0,:])):
        #varph,info=gmres(slp_strong,bs[:,j],maxiter=300,tol=10**(-7))
        varph_gf = bempp.api.linalg.lu(slp,bgrids[j],lu_factor=lufactors)
        #if info != 0:
        #    print("info: ", info)
        #varph_gf = bempp.api.GridFunction(p1_space,coefficients=varph)
        uscat[:,j]=slp_pot*varph_gf
    return uscat
