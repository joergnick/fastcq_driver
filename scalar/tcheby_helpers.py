import numpy as np

def tchebycheff_nodes(a, b, N):
    return a + 0.5*(b-a)*(np.cos((2*np.arange(N))*np.pi/(2*N-2))+1)

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
        if j==0 or j==len(borders)-2 or (am_evals<= K):
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
        currpoint=currpointcheck
    #print("Total amount of used evaluations: ",count_evals,". Amount of intervals: "+str(len(borders))+". Polynomial degree: "+str(K)+".")
    return y[:,0],count_evals-1