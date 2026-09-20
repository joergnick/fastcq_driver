import numpy as np
import matplotlib.pyplot as plt
from rkmethods import Extrapolator,RKMethod
from tcheby_helpers import tchebycheff_nodes,tcheb_interpolation,lagrange_interpolation,piecewise_interpolation,piecewise_interpolation_nodes, piecewise_interpolation_from_evals

class Conv_Operator():
    m = 1
    tol=10**-14
    #tol=10**-20
    external_N   = -1
    external_rho = None
    external_L = None
    factor_laplace_evaluations= -1
    def __init__(self,apply_elliptic_operator,order=2):
        self.order=order
        self.delta=lambda zeta : self.char_functions(zeta,order)
        self.apply_elliptic_operator=apply_elliptic_operator
    def get_integration_parameters(self,N,T):
        tol=self.tol
        dt=(T*1.0)/N
        if self.external_L:
            L = self.external_L
        else:
            if self.external_N<0:
                self.external_N = N
            #L=max(4*int(N),1)
            if self.m == 2:
                L=2*int(N)
            elif self.m == 3:
                L = 2*int(N)
            else:
                L = 2*int(N)

            #L=max(4*int(N),1)
            ###################### BEST WORKING PARAMETERS KNOWN: 
            #L=4*int(N)
        #L=3.0/2*N
        #rho=tol**(1.0/(2*L))
        if self.external_rho:
            rho = self.external_rho
            print("WARNING, EXTERNAL RHO WAS SET.")
        else:
            rho=tol**(1.0/(2*N))
            #rho=tol**(1.0/(3*N))
        return L,dt,tol,rho

    def char_functions(self,zeta,order):
        if order==1:
            return 1-zeta
        if order==2:
            return (1-zeta)+0.5*(1-zeta)**2
            #return 1.5-2.0*zeta+0.5*zeta**2
        if order ==3:
            return (1-zeta)+0.5*(1-zeta)**2+1.0/3.0*(1-zeta)**3
        if order ==-1:
            return 2*(1-zeta)/(1+zeta)
        else:
            print("Multistep order not availible")

    def get_frequencies(self,N,T):
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        Unit_Roots=np.exp(-1j*2*np.pi*(np.linspace(0,L-1,L)/(L)))
        return rho*Unit_Roots

    def get_method_characteristics(self,method):
        rk = RKMethod(method,1)
        return rk.A,rk.b,rk.c,rk.m
       
    def format_rhs(self,rhs,m,first_value_is_t0):
        try:
            rhs[0,0]
            flag_1Dinput=False
        except:
            flag_1Dinput=True
            rhs_mat = np.zeros((1,len(rhs))) 
            rhs_mat[0,:] = rhs
            rhs = rhs_mat
        if first_value_is_t0:
            rhs = rhs[:,1:]
        N=int((len(rhs[0,:]))//m)
        return rhs,N,flag_1Dinput

    def apply_RKconvol(self,rhs,T,show_progress=True,method="RadauIIA-2",factor_laplace_evaluations=2,cutoff=10**(-15),prolonge_by = 0,external_rho = None,external_L = None,first_value_is_t0 = True,return_freqs = False):
        if external_rho:
            self.external_rho = external_rho
        if external_L:
            self.external_L = external_L
        m      = RKMethod(method,1).m
        self.m=m    
        [rhs,N,flag_1Dinput]=self.format_rhs(rhs,m,first_value_is_t0)
        tau = T*1.0/N ## tau stays the same after prolongation

        rk   = RKMethod(method,tau) ## rk also stays the same, as tau is equal.
        prolonged_flag = (prolonge_by>0)
        ### Prolongation
        if (N>10) and prolonged_flag : 
            extr = Extrapolator()
            rhs  = extr.prolonge_towards_0(rhs,prolonge_by,rk)    
            [rhs,N,flag_1Dinput]=self.format_rhs(rhs,m,first_value_is_t0)
        if return_freqs:
            freqs = []
        ## Step 1 ##
        self.factor_laplace_evaluations = max(1,factor_laplace_evaluations)
        [A_RK,b_RK,c_RK,m]=self.get_method_characteristics(method)
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        rhs_fft=1j*np.zeros((len(rhs[:,0]),m*L))

        for stageInd in range(m):
            rhs_fft[:,stageInd:m*L:m]=self.scale_fft(rhs[:,stageInd:m*N:m],N,T)
        #Initialising important parameters for the later stage  
        s_vect= rho * np.exp(-1j*2*np.pi*(np.linspace(0,L-1,L)/(L)))
        ## Step 2 ##
        normsRHS=np.ones(m*L)
        counter=0
        for j in range(0,m*L):
            normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
            if normsRHS[j]>cutoff:
                counter=counter+1
        if normsRHS[0]>10**200:
            raise ValueError("Fourier coefficients have exploded.")

        HalfL= int(np.ceil(float(L)/2.0))
        if show_progress:
            print("Amount of Systems needed: "+ str(counter))

        ##Timing the elliptic systems
        import time
        start=0
        end=0
        rhsStages=1j*np.zeros((len(rhs_fft[:,0]),m))
        for j in range(0,HalfL+1):
            deltaMatrix=np.linalg.inv(A_RK+s_vect[j]*1.0/(1-s_vect[j])*np.ones((m,1))*b_RK)
            deltaEigs,T =np.linalg.eig(deltaMatrix/dt)
            if return_freqs:
                freqs.append(deltaEigs)
            if np.linalg.cond(T)>10**6:
                print("CONDITION LARGE", np.linalg.cond(T))
            Tinv=np.linalg.inv(T)
            if show_progress:
                print("j:",j,"L:",L, "Time of previous iteration: " +str((end-start)/60), " MIN" )
            start=time.time()
            rhsStages =  np.matmul(rhs_fft[:,m*j:m*(j+1)],Tinv.T)
            if j==0:  ## Separating cases, since dof of target space is unknown.
                first_eval=self.apply_elliptic_operator(deltaEigs[0],rhsStages[:,0])
                dof_trgt_spc = len(first_eval)
                phi_hat=1j*np.zeros((dof_trgt_spc,m*L))
                lhsStages=1j*np.zeros((dof_trgt_spc,m))
                lhsStages[:,0]=first_eval
                for stageInd in range(1,m):
                    lhsStages[:,stageInd]=self.apply_elliptic_operator(deltaEigs[stageInd],rhsStages[:,stageInd])
            for stageInd in range(m):
                if np.linalg.norm(rhsStages[:,stageInd])>cutoff :
                    lhsStages[:,stageInd]=self.apply_elliptic_operator(deltaEigs[stageInd],rhsStages[:,stageInd])       

            phi_hat[:,m*j:m*(j+1)] = np.matmul(lhsStages,T.T)
            end=time.time()
        ## Mirroring the second part of the frequencies by complex conjugation
        for freqInd in range(HalfL,L):  
            for stageInd in range(m):
                phi_hat[:,freqInd*m+stageInd]=np.conj(phi_hat[:,m*(L-freqInd)+stageInd])

        ## Step 3, Postprocessing
        for stageInd in range(m):
            phi_hat[:,stageInd:m*L:m]=self.rescale_ifft(phi_hat[:,stageInd:m*L:m],N,T)
        phi_sol=phi_hat[:,:m*N]
        if prolonged_flag:
            phi_sol =  extr.cut_back(phi_sol)
        if first_value_is_t0:
            phi_sol = np.concatenate((np.zeros((len(phi_sol[:,0]),1)),phi_sol ),axis = 1)
        if flag_1Dinput:
            phi_sol=phi_sol[0,:]
        if return_freqs:
            return phi_sol, freqs
        return phi_sol

    def apply_RKconvolfast(self,rhs,T,K,order_cq = 4,parsimonious=True,show_progress=True,method="RadauIIA-2",factor_laplace_evaluations=1,cutoff=10**(-15),prolonge_by = 0,external_rho = None,external_L = None,first_value_is_t0 = True,return_freqs = False):
        if external_rho:
            self.external_rho = external_rho
        if external_L:
            self.external_L = external_L
        m      = RKMethod(method,1).m
        self.m=m    
        [rhs,N,flag_1Dinput]=self.format_rhs(rhs,m,first_value_is_t0)
        tau = T*1.0/N ## tau stays the same after prolongation
        #self.tol = max(1e-14, 0.1*tau**4)


        ## Step 1 ##
        self.factor_laplace_evaluations = max(1,factor_laplace_evaluations)
        [A_RK,b_RK,c_RK,m]=self.get_method_characteristics(method)
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        rhs_fft=1j*np.zeros((len(rhs[:,0]),m*L))
         
        for stageInd in range(m):
            rhs_fft[:,stageInd:m*L:m]=self.scale_fft(rhs[:,stageInd:m*N:m],N,T)
        ## Step 2 ##
        normsRHS=np.ones(m*L)
        counter=0
        for j in range(0,m*L):
            normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
            if normsRHS[j]>cutoff:
                counter=counter+1
        if normsRHS[0]>10**200:
            raise ValueError("Fourier coefficients have exploded.")

        HalfL= int(np.ceil(float(L)/2.0))
        if show_progress:
            print("Amount of Systems needed: "+ str(counter))

        ##Timing the elliptic systems
        import time
        start=0
        end=0
        rhsStages=1j*np.zeros((len(rhs_fft[:,0]),m))
        
        ## Exact evaluation of the frequencies, no interpolation, no extrapolation.
        freqs = []
        Kvals = []
        branches = []
        lls = np.linspace(0,L-1,L)/(L)
        s_vect= rho * np.exp(-1j*2*np.pi*lls)
        ## Make frequencies
        for j in range(0,L):
            deltaMatrix=np.linalg.inv(A_RK+s_vect[j]*1.0/(1-s_vect[j])*np.ones((m,1))*b_RK)
            deltaEigs,T =np.linalg.eig(deltaMatrix/dt)
            Kvals.append(K(deltaEigs))
            freqs.append(deltaEigs)
            if j==0:
                if m == 2:
                    branches.append([0,1])
                if m == 3:
                    branches.append([0,1,2])
            if j>0:
                curr_branch=[]
                for stageInd in range(m):
                    curr_branch.append(np.argmin(np.abs(deltaEigs[stageInd]-freqs[j-1][branches[-1]])))
                branches.append(curr_branch)
        #for j in range(1,HalfL):
        #    Kvals.append(np.conj(Kvals[HalfL+1-j]))
        #    freqs.append(np.conj(freqs[HalfL+1-j]))
        Kvals = np.array(Kvals)
        ## Evaluation of the frequencies at interpolation points
        if m == 2:
            d = int(10*np.log(N)+10)
            cutoff_interp=0.2
            borders_per_branch = [[ cutoff_interp,0.25, 1],[0,0.75,1-cutoff_interp]]
        if m == 3:
            d = int(15*np.log(N)-30)
            cutoff_interp=0.34
            borders_per_branch = [[ 0, 1],[cutoff_interp,0.39,0.5, 1],[0,0.5,0.61,1-cutoff_interp]]
            #cutoff_interp=0.4
            #borders_per_branch = [[ 0, 1],[cutoff_interp,0.5, 1],[0,0.5,1-cutoff_interp]]
        #borders_per_branch = [[ cutoff_interp,0.12,0.15,0.2, 1],[0,0.8,0.85,0.88,1-cutoff_interp]]
        count_evals = 0
        #borders_per_branch = [[0, 0.2, 0.8, 1] for _ in range(m)]
        Kvals_tcheby_sorted = np.array([Kvals[j, branches[j]] for j in range(L)]).T
        for si in range(m):
            borders_si = borders_per_branch[si]
            prev_eig_si = None
            for piece in range(len(borders_si)-1):
                a, b = borders_si[piece], borders_si[piece+1]
                tch = np.sort(tchebycheff_nodes(a, b, d))
                s_tch = rho * np.exp(-1j * 2 * np.pi * tch)
                p_eigs, p_Kvals_si, p_idx = [], [], []
                for k in range(len(tch)):
                    deltaMatrix = np.linalg.inv(A_RK+s_tch[k]*1.0/(1-s_tch[k])*np.ones((m,1))*b_RK)
                    deltaEigs, _ = np.linalg.eig(deltaMatrix / dt)
                    if prev_eig_si is None and k == 0:
                        idx = si
                    elif k == 0:
                        idx = np.argmin(np.abs(deltaEigs - prev_eig_si))
                    else:
                        idx = np.argmin(np.abs(deltaEigs - p_eigs[-1][p_idx[-1]]))
                    p_idx.append(idx)
                    p_eigs.append(deltaEigs)
                    p_Kvals_si.append(K(deltaEigs)[idx])
                    count_evals += 1
                prev_eig_si = p_eigs[-1][p_idx[-1]]
                is_last = (piece == len(borders_si)-2)
                mask = (lls >= a) & (lls <= b) if is_last else (lls >= a) & (lls < b)
                lls_piece = lls[mask]
                if len(lls_piece) > 0:
                    nodes_scaled = 2*(tch       - a)/(b-a) - 1
                    lls_scaled   = 2*(lls_piece - a)/(b-a) - 1
                    Kvals_tcheby_sorted[si, mask] = tcheb_interpolation(nodes_scaled, np.array(p_Kvals_si), lls_scaled)

        count_evals += m*sum(lls<=cutoff_interp) 
        if count_evals >=L*m:
            Kvals_tcheby=Kvals
            count_evals = L*m
        else:
            ### Careful: ORDERING WRONG FOR T, generally not working
            Kvals_tcheby= np.array([Kvals_tcheby_sorted[branches[j],j] for j in range(len(branches))])
        print("Fast evaluations: ", cutoff_interp*L*m+2*len(borders_per_branch[0])*d,
              "Ex metric:",count_evals, "Frequencies before cutoff:",cutoff_interp*L*m, " Number of frequencies: ",L*m)
        plt.figure(0)
        #plt.subplot(1,2,1)
        Kbranches = [Kvals[j,branches[j]] for j in range(len(branches))]
        plt.semilogy(np.linspace(0,1,len(Kbranches)),np.abs(Kvals_tcheby_sorted.T-Kbranches))
        plt.savefig('Kvals_errs.pdf')
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.plot(np.real(Kvals_tcheby_sorted).T)
        plt.plot(np.real(Kbranches),linestyle='dashed')
        plt.subplot(1,2,2)
        plt.plot(np.imag(Kvals_tcheby_sorted).T)
        plt.plot(np.imag(Kbranches),linestyle='dashed')
        plt.savefig("Kvals.pdf")
        #print("Error: ",np.abs(Kvals-Kvals_tcheby))
        print("Max error: ",np.max(np.abs(Kvals-Kvals_tcheby)))
        for j in range(0,HalfL+1):
            deltaMatrix=np.linalg.inv(A_RK+s_vect[j]*1.0/(1-s_vect[j])*np.ones((m,1))*b_RK)
            deltaEigs,T =np.linalg.eig(deltaMatrix/dt)
            if return_freqs:
                freqs.append(deltaEigs)
            if np.linalg.cond(T)>10**6:
                print("CONDITION LARGE", np.linalg.cond(T))
            Tinv=np.linalg.inv(T)
            if show_progress:
                print("j:",j,"L:",L, "Time of previous iteration: " +str((end-start)/60), " MIN" )
            start=time.time()
            rhsStages =  np.matmul(rhs_fft[:,m*j:m*(j+1)],Tinv.T)
            if j==0:  ## Separating cases, since dof of target space is unknown.
                if parsimonious:
                    first_eval=Kvals_tcheby[0,0]*rhsStages[:,0]
                else:
                    first_eval=Kvals[0,0]*rhsStages[:,0]
                #first_eval=self.apply_elliptic_operator(deltaEigs[0],rhsStages[:,0])
                dof_trgt_spc = len(first_eval)
                phi_hat=1j*np.zeros((dof_trgt_spc,m*L))
                lhsStages=1j*np.zeros((dof_trgt_spc,m))
                lhsStages[:,0]=first_eval
                for stageInd in range(1,m):
                    if parsimonious:
                        lhsStages[:,stageInd]=Kvals_tcheby[j,stageInd]*rhsStages[:,stageInd]
                    else:
                        lhsStages[:,stageInd]=Kvals[j,stageInd]*rhsStages[:,stageInd]
                    #lhsStages[:,stageInd]=self.apply_elliptic_operator(deltaEigs[stageInd],rhsStages[:,stageInd])
            for stageInd in range(m):
                if np.linalg.norm(rhsStages[:,stageInd])>cutoff :
                    if parsimonious:
                        lhsStages[:,stageInd]=Kvals_tcheby[j,stageInd]*rhsStages[:,stageInd]
                    else:
                        lhsStages[:,stageInd]=Kvals[j,stageInd]*rhsStages[:,stageInd]
 
                    #lhsStages[:,stageInd]=Kvals[j,stageInd]*rhsStages[:,stageInd]       

            phi_hat[:,m*j:m*(j+1)] = np.matmul(lhsStages,T.T)
            end=time.time()
        ## Mirroring the second part of the frequencies by complex conjugation
        for freqInd in range(HalfL,L):  
            for stageInd in range(m):
                phi_hat[:,freqInd*m+stageInd]=np.conj(phi_hat[:,m*(L-freqInd)+stageInd])

        ## Step 3, Postprocessing
        for stageInd in range(m):
            phi_hat[:,stageInd:m*L:m]=self.rescale_ifft(phi_hat[:,stageInd:m*L:m],N,T)
        phi_sol=phi_hat[:,:m*N]
        if first_value_is_t0:
            phi_sol = np.concatenate((np.zeros((len(phi_sol[:,0]),1)),phi_sol ),axis = 1)
        if flag_1Dinput:
            phi_sol=phi_sol[0,:]
        if return_freqs:
            return phi_sol, freqs,count_evals
        return phi_sol,count_evals


    def get_zeta_vect(self,N,T):
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        import numpy as np
        Unit_Roots=np.exp(-1j*2*np.pi*(np.linspace(0,L,L+1)/(L+1)))
        Zeta_vect=self.delta( rho* Unit_Roots)/dt 
        return Zeta_vect




    def gen_RK_frequencies(self,N,T,show_progress=False,method="RadauIIA-2"):
        m      = RKMethod(method,1).m
        self.m=m    
        tau = T*1.0/N ## tau stays the same after prolongation

        rk   = RKMethod(method,tau) ## rk also stays the same, as tau is equal.
        ### Prolongation
        freqs = []
        ## Step 1 ##
        self.factor_laplace_evaluations = 2
        [A_RK,b_RK,c_RK,m]=self.get_method_characteristics(method)
        L,dt,tol,rho=self.get_integration_parameters(N,T)

        #Initialising important parameters for the later stage  
        s_vect= rho * np.exp(-1j*2*np.pi*(np.linspace(0,L-1,L)/(L)))
        ## Step 2 ##
        normsRHS=np.ones(m*L)
        counter=0

        if show_progress:
            print("Amount of Systems needed: "+ str(counter))

        ##Timing the elliptic systems
        import time
        start=0
        end=0
        for j in range(0,L):
            deltaMatrix=np.linalg.inv(A_RK+s_vect[j]*1.0/(1-s_vect[j])*np.ones((m,1))*b_RK)
            deltaEigs,T =np.linalg.eig(deltaMatrix/dt)
            
            freqs.append(deltaEigs)

        return freqs

    def apply_convol(self,rhs,T,show_progress=False,method="BDF2",cutoff=10**(-8)):
        ## Step 1
        if not len(rhs[0]):
            raise ValueError("Right hand side provided is empty.")
        try:
            N=len(rhs[0,:])-1
        except:
            N=len(rhs)-1    
            rhs_mat=np.zeros((1,N+1))
            rhs_mat[0,:]=rhs
            rhs=rhs_mat

        L,dt,tol,rho=self.get_integration_parameters(N,T)
        normsRHS=np.ones(N+1)
        for j in range(0,N+1):
            normsRHS[j]=np.max(np.abs(rhs[:,j]))
####################### Scaling +FFT of RHS ##################################  
        n_rows=len(rhs[:,0])
        n_columns=len(rhs[0,:])
        rhs=rho**(np.linspace(0,n_columns-1,N+1))*rhs
        rhs_fft=np.fft.fft(np.concatenate((rhs,np.zeros((n_rows,L+1-n_columns))),axis=1))
#######################     
        #Initialising important parameters for the later stage  
        Zeta_vect=self.get_zeta_vect(N,T)
        dof=len(rhs[:,0])
        Half=int(np.ceil(float(L)/2.0))
        ## Step 2
        #phi_hat=1j*np.zeros((dof,L+1))
        normsRHS=np.ones(L+1)
        counter=0
        for j in range(0,Half+1):
            normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
            if normsRHS[j]>cutoff:
                counter=counter+1
        #Timing the elliptic systems
        import time
        start=0
        end=0
        first_eval=self.apply_elliptic_operator(Zeta_vect[0],rhs_fft[:,0])
        phi_hat=1j*np.zeros((len(first_eval),L+1))
        phi_hat[:,0]=first_eval

        for j in range(1,Half+1):
            normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
            #print("normRHS:",normsRHS[j])
            if normsRHS[j]>cutoff:
                if show_progress:
                    print("j:",j,"L:",str(Half), "Time of previous iteration: " +str((end-start)/60), " MIN" )
                start=time.time()
                phi_hat[:,j]=self.apply_elliptic_operator(Zeta_vect[j],rhs_fft[:,j])
                end=time.time()
        for j in range(Half+1,L+1):
            phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])      
        ## Step 3

        phi_sol=self.rescale_ifft(phi_hat,N,T)
        phi_sol=phi_sol[:,:N+1]
        return phi_sol


    def scale_fft(self,A,N,T):
        import numpy as np
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        n_rows=len(A[:,0])
        n_columns=len(A[0,:])
        A=rho**(np.linspace(0,n_columns-1,n_columns))*A
        A_fft=np.fft.fft(np.concatenate((A,np.zeros((n_rows,L-n_columns))),axis=1))
        return A_fft

    def rescale_ifft(self,A,N,T):
        import numpy as np
        L,dt,tol,rho=self.get_integration_parameters(N,T)
        n_rows=len(A[:,0])
        n_columns=len(A[0,:])

        A_ift=np.real(np.fft.ifft(A))
        A_sol=rho**(-np.linspace(0,n_columns-1,n_columns))*A_ift[:,0:n_columns]
        return(A_sol)

