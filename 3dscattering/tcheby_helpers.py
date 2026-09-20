
import numpy as np

def tchebycheff_nodes(a, b, N):
    #Tchebycheff nodes of the first kind
    #return 0.5*(a+b) + 0.5*(b-a)*(np.cos((2*np.arange(N)+1)*np.pi/(2*N)))
    #Tchebycheff nodes of the second kind:
    return a + 0.5*(b-a)*(np.cos((2*np.arange(N))*np.pi/(2*N-2))+1)
    #return 0.5*(a+b) + 0.5*(b-a)*np.cos((2*np.arange(N))*np.pi/(2*N-2))
#a = -1
#b = 1
#N = 8
#x_old = a + 0.5*(b-a)*(np.cos((2*np.arange(N))*np.pi/(2*N-2))+1)
#firstk = np.cos((2*np.arange(N)+1)/(2*N)*np.pi)
#seck   = np.cos(np.arange(N)/(N-1)*np.pi)
#x_new = 0.5*(a+b) + 0.5*(b-a)*(np.cos((2*np.arange(N)+1)*np.pi/(2*N)))
#print("First kind: ",firstk)
#print("Second kind: ",seck)
#print(" ")
#print("xold: ",x_old)
#print("xnew: ",x_new)
def lagrange_basis(x_nodes, x):
    x_nodes = np.asarray(x_nodes)
    x = np.asarray(x)
    N = len(x_nodes)
    L = np.ones((N, len(x)))
    L2 = [[]]
    for i in range(N):
        temp = [(x - x_nodes[j]) / (x_nodes[i] - x_nodes[j]) for j in range(N) if i != j]
        for t in temp:
            L[i, :] *= t
    return L

#def tcheb_interpolation(x_nodes, y_nodes, x):
#    """
#    Tchebycheff interpolation function.
#
#    Parameters:
#    x_nodes (numpy.ndarray): The x-coordinates of the nodes.
#    y_nodes (numpy.ndarray): The y-coordinates of the nodes.
#    x (float): The x-coordinate at which to evaluate the interpolating polynomial.
#
#    Returns:
#    float: The value of the interpolating polynomial at x.
#    """
#    n = len(x_nodes)
#    assert len(y_nodes) == n, "x_nodes and y_nodes must have the same length"
#
#    # Compute the Chebyshev-Vandermonde matrix
#    T = np.zeros((n, n))
#    T[:,0] = 1
#    if n > 1:
#        T[:,1] = x_nodes
#    for k in range(2, n):
#        T[:,k] = 2 * x_nodes * T[:,k-1] - T[:,k-2]
#
#    # Solve the linear system to get the coefficients of the interpolating polynomial
#    from scipy.linalg import solve
#    c = solve(T, y_nodes)
#
#    # Evaluate the interpolating polynomial at x
#    y = c[0]
#    T_x_prev = 1
#    T_x = x
#    for k in range(1, n):
#        y += c[k] * T_x
#        T_x_next = 2 * x * T_x - T_x_prev
#        T_x_prev = T_x
#        T_x = T_x_next
#
#    return y

def tcheb_interpolation(x_nodes, y_nodes, x):
    """
    Tchebycheff interpolation function.

    Parameters:
    x_nodes (numpy.ndarray): The x-coordinates of the nodes.
    y_nodes (numpy.ndarray): The y-coordinates of the nodes.
    x (float): The x-coordinate at which to evaluate the interpolating polynomial.

    Returns:
    float: The value of the interpolating polynomial at x.
    """
    n = len(x_nodes)
    assert len(y_nodes) == n, "x_nodes and y_nodes must have the same length"

    # Compute the Chebyshev coefficients
    c = np.polynomial.chebyshev.chebfit(x_nodes, y_nodes, n-1)

    # Use the Clenshaw algorithm to evaluate the interpolating polynomial
    b_kp2 = b_kp1 = 0
    for k in range(n-1, -1, -1):
        b_k = c[k] + 2*x*b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k

    return b_k - x*b_kp2
def lagrange_interpolation(x_nodes, y_nodes, x):
    L = lagrange_basis(x_nodes, x)
    return np.dot(y_nodes, L)

def piecewise_interpolation(borders,degrees,evalpoints,f):
    y = 1j*np.zeros((len(evalpoints),1))
    currpoint = 0
    count_evals = 0
    for j in range(len(borders)-1):
        a = borders[j]
        b = borders[j+1]
        K = degrees[j]
        ## Check amont of nodes in this interval:
        currpointcheck =currpoint
        while evalpoints[currpointcheck] <= borders[j+1]: 
            currpointcheck +=1
            if currpointcheck == len(evalpoints):
                break
        am_evals = currpointcheck-currpoint
        #if (am_evals<= K):
        if j==0 or j==len(borders)-2 or (am_evals<= K):
        #if j==0 or j==len(borders)-2 or (am_evals<= K):
            while evalpoints[currpoint] <= borders[j+1]: 
                y[currpoint] = f(evalpoints[currpoint])
                currpoint +=1
                count_evals +=1
                if currpoint == len(evalpoints):
                    break
            continue
        # Compute the Tchebycheff nodes
        x_nodes = tchebycheff_nodes(a, b, K)
        # Compute the function values at the nodes
        y_nodes = f(x_nodes)
        count_evals +=int(degrees[j])
        y[currpoint:currpointcheck+1,0] = tcheb_interpolation(x_nodes, y_nodes, evalpoints[currpoint:currpointcheck+1])
        
        #y[currpoint:currpointcheck+1,0] = lagrange_interpolation(x_nodes, y_nodes, evalpoints[currpoint:currpointcheck+1])
        #yvals= lagrange_interpolation(x_nodes, y_nodes, evalpoints[currpoint:currpointcheck+1])
        currpoint=currpointcheck
        #while evalpoints[currpoint] <= borders[j+1]: 
        #    # Compute the interpolation
        #    y[currpoint] = lagrange_interpolation(x_nodes, y_nodes, [evalpoints[currpoint]])[0]
        #    print(y[currpoint])
        #    currpoint +=1
        #    if currpoint == len(evalpoints):
        #        break
    print("Total amount of used evaluations: ",count_evals,". Amount of intervals: "+str(len(borders))+". Polynomial degree: "+str(K)+".")
    return y[:,0],count_evals-1