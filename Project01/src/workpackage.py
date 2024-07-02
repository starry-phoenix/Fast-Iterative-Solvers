import numpy as np
import numpy.linalg as la 
from typing import IO

def MSR_matmul(JM,VM,X, symm = False):
    """ Matrix vector multiplication for sparse positive definite matrices stored in MSR format
    
    Ax = y

    Parameters
    ----------
    JM : array
        MSR column index array of matrix A
    VM : array
        MSR value array of matrix A
    X : array
        vector
    symm : bool 
        True if symmetric matrix
    Returns
    -------
    YM : array
        result of matrix vector multiplication
    
    Example:
    -------
    J_M = np.array([6,7,8,8,9,3,3,3])   # Matrix index for 4x4 matrix
    VM = np.array([1,2,1,2,0,4,2,4])   # Matrix values for 4x4 matrix
    X = np.array([1,1,1,1])

    MSR_matmul(JM=J_M,VM=VM,X=X)
    # Output: array([ 5.,  4.,  1.,  0.])
    """

    n_d = np.where(VM==0)[0][0]                 # number of diagonal elements until the pointer = 0
    V_d = VM[:n_d]                              # until the pointer = diagonal elements
    V_nd = VM[n_d+1:]                           # after the pointer = non-diagonal elements
    IA_M = JM[:n_d+1]                           # CSR row pointer for non-diagonal elements
    J_M = JM[n_d+1:]                            # CSR column index for non-diagonal elements
    YM = np.zeros((n_d))

    for i in range(n_d):                        # diagonal elements
        YM[i] = V_d[i]*X[i]

    n = IA_M.shape[0]-1                         # number of rows
    IA_M = IA_M - (n_d+1)                       # Shifts row pointer to start from 1 instead of n+1
    for i in range(n):                          # row index
        i1 = IA_M[i]-1                          # row index start
        i2 = IA_M[i+1]-1                        # row index end
        for j in range(i1,i2):
            YM[i] += V_nd[j]*X[J_M[j]-1]        # Update the YM array, YM[i] += non-diagonal elements * X[column index]
            if symm and (J_M[j]-1) != i:        # If symmetric matrix and column index is not equal to row index
                YM[J_M[j]-1] += V_nd[j]*X[i]    # Update the YM array YM[column index] += non-diagonal elements * X[row index]
            else:
                pass

    return YM

def get_M_msr(JM, VM, symm:bool=False, precond=None):
    """ Get M matrix for Jacobi/ Gauss Seidel preconditioner for matrices stored in MSR format
    Parameters
    ----------
    JM : array
        MSR column index array
    VM : array
        MSR value array
    symm : bool
        symmetric matrix
    precond : str
        preconditioning type
        'Jacobi' or 'GS'
    Returns
    ------- 
    M : array
        preconditioning matrix
    """
    if precond == 'Jacobi':
        n_d = np.where(VM==0)[0][0]  # number of diagonal elements
        V_d = VM[:n_d]               # until the pointer = diagonal elements
        M = np.zeros((n_d))
        for i in range(n_d):        # diagonal elements
            M[i] = V_d[i]           
    elif precond == 'GS':
        n_d = np.where(VM==0)[0][0]  # number of diagonal elements
        V_d = VM[:n_d]               # until the pointer = diagonal elements
        M = np.zeros((n_d,n_d))
        for i in range(n_d):
            M[i][i] = V_d[i]        # diagonal elements
        V_nd, IA_M, J_M = VM[n_d+1:], JM[:n_d+1], JM[n_d+1:]    # non-diagonal elements, row pointer, column index
        n = IA_M.shape[0]-1
        IA_M = IA_M - (n_d+1)                                   
        for i in range(n):                                      # Loops over the row index to store lower triangular elements
            for j in range(IA_M[i], IA_M[i+1]):
                if i > J_M[j-1]-1:
                    M[i][J_M[j-1]-1] = V_nd[j-1]
    elif precond is None:
        n_d = np.where(VM==0)[0][0]  # number of diagonal elements
        M  = np.identity(n_d)
    else:
        print("invalid preconditioning")

    return M

def norm(A):
    """ Calculate the norm of a vector A """
    sum = 0
    for i in range(A.shape[0]):
        sum += A[i]**2

    return np.sqrt(sum)

def dot_product(A,B):
    """ Calculate the dot product of two vectors A and B """
    sum = 0
    for i in range(A.shape[0]):
        sum += A[i]*B[i]

    return sum

def matrix_vector_product(A,v):
    """ Matrix vector multiplication for matrix A and vector v """
    n = A.shape[0]  #v.shape[0]
    w = np.zeros((n))
    for i in range(n):
        w[i] = dot_product(A[i],v)

    return w

def get_krylov_msr(A_JM, A_VM, v_full_j, j, h_j, M_inv,precond=None, symm=False):
    """ Get the Krylov subspace for matrices stored in MSR format 
    
    Parameters
    ----------
    A_JM : array
        MSR column index array
    A_VM : array
        MSR value array
    v_full_j : array
        Krylov vectors (orthonormal basis) until iteration j
    j : int
        iteration number
    h_j : array
        hessenberg matrix
    M_inv : array
        inverse of preconditioning matrix M
    precond : str
        preconditioning type
        'Jacobi' or 'GS'    
    symm : bool
        symmetric matrix
    Returns
    -------
    h_j : array
        hessenberg matrix
    v_full_j : array
        orthonormal basis
    v_j : array
        orthonormal basis

    """
    w = MSR_matmul(A_JM, A_VM,v_full_j[:,j], symm=symm)           # matrix_vector_product A*V where V = (v1,v2,v3,..,vj)
    w  = apply_preconditioner_msr(M_inv, w, precond=precond)      # M_inv * (A*V)
    for i in range(j+1):
        h_j[i][j] = dot_product(w,v_full_j[:,i])
        w = w - h_j[i][j]*v_full_j[:,i]
    h_j[j+1][j] = norm(w)
    v_full_j[:,j+1] = w/h_j[j+1][j]
    v_j = v_full_j[:,j+1]
    return h_j,v_full_j,v_j

def back_substitution(h,g, m):
    """ Back substitution for solving the linear system Ax = b after the Krylov subspace is obtained

    Parameters
    ----------
    h : array
        hessenberg matrix
    g : array
        g = norm(r0)
    m : int
        iteration number
    Returns
    -------
    y : array
        solution of the linear system Ax = b
    """
    m = m -1
    y = np.zeros((m+1))
    y[m] = g[m]/h[m][m]
    for i in range(m-1,-1,-1):
        y[i] = g[i]
        for j in range(i+1,m+1):
            y[i] = y[i] - h[i][j]*y[j]
        y[i] = y[i]/h[i][i]

    return y

def forward_substitution(h,g):
    """ Forward substitution for matrix inversion using Gauss Seidel preconditioning
    Linear system y = H^-1 g where H is the lower triangular matrix
    """
    m = h.shape[0]
    y = np.zeros((m))
    y[0] = g[0]/h[0][0]
    for i in range(1,m):
        y[i] = g[i]
        for j in range(i):
            y[i] = y[i] - h[i][j]*y[j]
        y[i] = y[i]/h[i][i]

    return y

def lower_matrix(A):
    """ Get the lower triangular matrix from A of size m x m"""
    m = A.shape[0]
    L = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1):
            L[i][j] = A[i][j]

    return L

def apply_preconditioner_msr(M_inv, matrix, precond=None):
    if precond == 'Jacobi':
        return M_inv * matrix
    elif precond == 'GS':
        #print('Using Gauss Seidel preconditioning')
        return forward_substitution(M_inv, matrix)
    else:
        #print('No preconditioning')
        return matrix
    
def return_Minv_msr(M, precond=None):
    # M is the preconditioning matrix to be inverted
    if precond == 'Jacobi':                         # Jacobi preconditioning
        n_d = len(M)
        Minv = np.zeros((n_d))
        for i in range(n_d):
            Minv[i] = 1.0/M[i]                      # Inverse of the diagonal elements
    elif precond == 'GS':                           # Gauss Seidel preconditioning
        Minv = M                                    # Performs the inversion in the forward substitution
    else:
        Minv = np.identity(M.shape[0])
    
    return Minv

def gmres_msr(A_JM, A_VM,b, x_0, M=None, maxiter=None, tol=1e-08, precond=None,file: IO = None, symm=False, rho_0=1.0, Minv=None):
    """" 
    GMRES for matrices stored in MSR format
    Ax = b
    Parameters
    ----------
    A_JM : array
        MSR column index array of matrix A
    A_VM : array
        MSR value array of matrix A
    b : array
        RHS vector
    x_0 : array
        initial guess
    M : array
        preconditioning matrix
    maxiter : int
        maximum number of iterations
    tol : float
        tolerance
    precond : str
        preconditioning type
        'Jacobi' or 'GS'
    file : IO
        file to write the residuals
    symm : bool
        symmetric matrix
    rho_0 : float
        initial residual
    Minv : array
        inverse of preconditioning matrix
    Returns
    -------
    x : array
        solution
    rho : float
        residual
    """
    m = b.shape[0]
    if precond is None: M = np.identity(m)
    M_inv = Minv                                                #return_Minv_msr(M, precond)
    break_cond = tol
    v_norm = r_0/norm(r_0)                                      # normalize r0
    h = np.zeros((m+1,m))                                       # hessenberg matrix
    v_full = np.zeros((m,m+1))                                  # orthonormal basis
    v_full[:, 0] = v_norm                                       # Krlov vector full matrix V
    c, s = [],[]                                                # Givens rotation vectors c and s for cosines and sines
    g = [norm(r_0)]                                             # stores givens rotation vectors

    r_0 = b - MSR_matmul(A_JM, A_VM, x_0, symm=symm)            # r0 = b - Ax0
    r_0 = apply_preconditioner_msr(M_inv, r_0, precond)         # r0 = M^-1 * (b - Ax0)  # Preconditioning

    if maxiter is None: maxiter = m 

    # iterate over the restart parameter (maxiter = restart parameter)
    for j in range(maxiter):
        # get the Krylov subspace for the matrix A
        # h = hessenberg matrix, v_full = orthonormal basis, vp1 = orthonormal basis
        h,v_full,vp1 = get_krylov_msr(A_JM, A_VM,v_full, j, h, M_inv, precond, symm=symm)
        # apply the Givens rotation to the hessenberg matrix
        for k in range(1,j+1):
            h1 = c[k-1]*h[k-1][j] + s[k-1]*h[k][j]
            h2 = c[k-1]*h[k][j] - s[k-1]*h[k-1][j]
            h[k-1][j] = h1
            h[k][j] = h2
        alpha = np.sqrt(h[j][j]**2 + h[j+1][j]**2)
        c.append(h[j][j]/alpha)
        s.append(h[j+1][j]/alpha)
        h[j][j] = c[j]*h[j][j] + s[j]*h[j+1][j]
        h[j+1][j] = 0
        g.append(-s[j]*g[j])                                # append the givens rotation vectors to g
        g[j] = c[j]*g[j]                                   
        rho = abs(g[j+1])                                   # absolute residual
        rel_res = rho/abs(norm(rho_0))                      # relative residual
        file.write(f"{j} , {rho}\n")
        if rho < break_cond:
            break

    m_bar = min(m,j+1)
    y = back_substitution(h[:m_bar+1][:m_bar],g[:m_bar+1],m_bar)    # back substitution to solve the linear system y = H^-1 g
    x = x_0 + matrix_vector_product(v_full[:,:m_bar],y)             # solution of the linear system Ax = b: x = x0 + V*y
    
    return x,rho

def restart_gmres_msr(A_JM, A_VM,b,x_0,M,m,tol,precond, symm=False, max_while_count=500):
    """ Restart GMRES for matrices stored in MSR format
    
    Parameters
    ----------
    A_JM : array
        The Jacobian matrix in MSR format.
    A_VM : array
        The value matrix in MSR format.
    b : array
        The right-hand side vector.
    x_0 : array
        The initial guess for the solution.
    M : array
        The preconditioner matrix.
    m : int
        The number of iterations to restart GMRES.
    tol : float
        The tolerance for convergence.
    precond : str
        The type of preconditioner to be used.
    symm : bool, optional
        Whether the matrix is symmetric or not. Default is False.
    max_while_count : int, optional
        The maximum number of iterations for GMRES. Default is 500.
    
    Returns
    -------
    x_m : array
        The solution vector obtained from GMRES.
    
    Example
    -------
    >>> A_JM = np.array([6,7,8,8,9,3,3,3])
    >>> A_VM = np.array([1,2,1,2,0,4,2,4])
    >>> b = np.array([1,1,1,1])
    >>> x_0 = np.array([1,1,1,1])
    >>> M = get_M_msr(A_JM, A_VM)
    >>> x_m = restart_gmres_msr(A_JM, A_VM, b, x_0, M)
    """

    r = b - MSR_matmul(A_JM, A_VM, x_0, symm=symm)                  # r = b - Ax0
    M_inv = return_Minv_msr(M, precond)                             # M_inv = inverse of preconditioning matrix
    r = apply_preconditioner_msr(M_inv, r, precond)                 # r = M^-1 * (b - Ax0)  # Preconditioning
    rho = norm(r)/norm(r)                                           # initial residual
    # loops until it converges to the tolerance
    with open("gmres_residuals_" + str(m) +".txt", "w") as file:
        file.write("Iter , rho\n")
        x = x_0
        count = 0
        while rho > tol:
            x_m, rho = gmres_msr(A_JM, A_VM, b, x, M, m, tol, precond, file, symm=symm, rho_0=r, Minv = M_inv)
            x = x_m
            count +=1
            if count > max_while_count:
                print("GMRES did not converge in {} iterations".format(max_while_count))
                break
            elif rho < tol:
                print("GMRES converged in {} iterations".format(count))
            else:
                pass

    print("Final residual: ", rho)
    print('Final relative residual: ', rho/norm(r))

    return x_m, ('gmres_residuals_' + str(m) + '.txt')

def err_arnoldi(v_arnoldi, h_arnoldi, a_matrix):
    return la.norm(v_arnoldi[:,:-1].T @ a_matrix @ v_arnoldi[:, :-1] - h_arnoldi[:-1])/la.norm(a_matrix)

def conjugate_gradient_msr(A_JM, A_VM, b_, x0_cg, tol=1e-8, maxiter=10, symm=True):
    """Conjugate gradient method for solving linear systems with matrices stored in MSR format.

    Parameters:
    - A_JM (ndarray): The MSR format matrix J.
    - A_VM (ndarray): The MSR format matrix V.
    - b_ (ndarray): The right-hand side vector.
    - x0_cg (ndarray): The initial guess for the solution.
    - tol (float, optional): The tolerance for convergence. Defaults to 1e-8.
    - maxiter (int, optional): The maximum number of iterations. Defaults to 10.
    - symm (bool, optional): Flag indicating whether the matrix is symmetric. Defaults to True.

    Returns:
    - ndarray: The solution vector.

    Raises:
    - None

    Notes:
    - This function implements the conjugate gradient method for solving linear systems with matrices stored in MSR format.
    - The MSR format consists of two arrays: J (the MSR indices array) and V (the MSR values array).
    - The function writes the residuals to a file named "cg_residuals.txt" in the current directory.
    - If the conjugate gradient method does not converge within the specified maximum number of iterations, a warning message is printed.

    Example:
    >>> A_JM = np.array([6,7,8,8,9,3,3,3])
    >>> A_VM = np.array([1,2,1,2,0,4,2,4])
    >>> b = np.array([1,1,1,1])
    >>> X = np.array([1,1,1,1])
    >>> x_cg = conjugate_gradient_msr(Jm_gmres, Vm_gmres, b, X)
    """
    r0 = b_ - MSR_matmul(A_JM, A_VM, x0_cg, symm=symm)
    p0 = r0
    xm = x0_cg
    rm = r0
    pm = p0
    x = np.ones(x0_cg.shape[0])
    r0_res = norm(MSR_matmul(A_JM, A_VM,x, symm=symm))
    with open("cg_residuals.txt", "w") as file:
        file.write("Iter,rho\n")    #,e_anorm\n")
        for i in range(maxiter):  # Changed from size to maxiter
            Apm = MSR_matmul(A_JM, A_VM, pm, symm=symm)
            alpha_m = np.dot(rm,rm) / np.dot(pm, Apm)
            xmp1 = xm + alpha_m * pm
            rmp1 = rm - alpha_m * Apm
            beta_m = np.dot(rmp1, rmp1) / np.dot(rm, rm)
            pmp1 = rmp1 + beta_m * pm
            xm = xmp1
            rm = rmp1
            pm = pmp1
            rel_res = np.linalg.norm(rmp1)/r0_res
            err_k = xmp1 - x
            file.write(f"{i},{rel_res}\n")    
            if np.linalg.norm(rmp1) < tol:  # Check for convergence
                break
            elif i == maxiter-1:
                print(f"Conjugate gradient did not converge in {maxiter} iterations")
                break 
            else:
                pass

    print(f"Conjugate gradient converged in {i} iterations")

    return xmp1, ('cg_residuals.txt')