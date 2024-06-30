import numpy as np
from scipy.sparse import csr_matrix
import numpy.linalg as la 
from typing import IO

def csr_test():
    IA = [1-1,3-1,5-1,6-1,8-1]
    J = [1-1,3-1,2-1,3-1,3-1,3-1,4-1]
    V = [1,4,2,2,1,4,2]
    csrm = csr_matrix((V,J,IA), shape=(4,4)).toarray()
    #print("CSR scipy is ".format(csr_matrix((V,J,IA), shape=(4,4))*X))

    return csrm

def MSR_matmul(JM,VM,X, symm = False):
    n_d = np.where(VM==0)[0][0]  # number of diagonal elements
    V_d = VM[:n_d]   # until the pointer = diagonal elements
    V_nd = VM[n_d+1:]  # after the pointer = non-diagonal elements
    IA_M = JM[:n_d+1]  # CSR row pointer for non-diagonal elements
    J_M = JM[n_d+1:]  # CSR column index for non-diagonal elements
    YM = np.zeros((n_d))

    for i in range(n_d):
        YM[i] = V_d[i]*X[i]

    n = IA_M.shape[0]-1
    IA_M = IA_M - (n_d+1)
    for i in range(n):   # row index
        i1 = IA_M[i]-1
        i2 = IA_M[i+1]-1
        for j in range(i1,i2):
            YM[i] += V_nd[j]*X[J_M[j]-1]
            if symm and (J_M[j]-1) != i:
                YM[J_M[j]-1] += V_nd[j]*X[i]   # Update the YM array by adding the product of V_nd[j] and X[i] to YM[J_M[j]-1]
            else:
                pass

    return YM

def get_M_msr(JM, VM, symm=False, precond=None):
    if precond == 'Jacobi':
        n_d = np.where(VM==0)[0][0]  # number of diagonal elements
        V_d = VM[:n_d]   # until the pointer = diagonal elements
        M = np.zeros((n_d))
        for i in range(n_d):
            M[i] = V_d[i]
    elif precond == 'GS':
        n_d = np.where(VM==0)[0][0]  # number of diagonal elements
        V_d = VM[:n_d]   # until the pointer = diagonal elements
        M = np.zeros((n_d,n_d))
        for i in range(n_d):
            M[i][i] = V_d[i]
        V_nd, IA_M, J_M = VM[n_d+1:], JM[:n_d+1], JM[n_d+1:]
        n = IA_M.shape[0]-1
        IA_M = IA_M - (n_d+1)
        for i in range(n):
            for j in range(IA_M[i], IA_M[i+1]):
                if i > J_M[j-1]-1:
                    M[i][J_M[j-1]-1] = V_nd[j-1]
    else:
        print("invalid preconditioning")

    return M


def norm(A):
    sum = 0
    for i in range(A.shape[0]):
        sum += A[i]**2

    return np.sqrt(sum)

def dot_product(A,B):
    sum = 0
    for i in range(A.shape[0]):
        sum += A[i]*B[i]

    return sum

def matrix_vector_product(A,v):
    n = A.shape[0]  #v.shape[0]
    w = np.zeros((n))
    for i in range(n):
        w[i] = dot_product(A[i],v)

    return w

def get_krylov(A, v_full_j, j, h_j, M_inv, precond=None):
    w = matrix_vector_product(A,v_full_j[:,j])
    w  = apply_preconditioner(M_inv, w, precond)      #M_inv @ w   # Preconditioning
    for i in range(j+1):
        h_j[i][j] = dot_product(w,v_full_j[:,i])
        w = w - h_j[i][j]*v_full_j[:,i]
    h_j[j+1][j] = norm(w)
    v_full_j[:,j+1] = w/h_j[j+1][j]
    v_j = v_full_j[:,j+1]
    return h_j,v_full_j,v_j

def get_krylov_msr(A_JM, A_VM, v_full_j, j, h_j, M_inv,precond=None, symm=False):
    w = MSR_matmul(A_JM, A_VM,v_full_j[:,j], symm=symm)
    w  = apply_preconditioner_msr(M_inv, w, precond=precond)      #M_inv @ w   # Preconditioning
    for i in range(j+1):
        h_j[i][j] = dot_product(w,v_full_j[:,i])
        w = w - h_j[i][j]*v_full_j[:,i]
    h_j[j+1][j] = norm(w)
    v_full_j[:,j+1] = w/h_j[j+1][j]
    v_j = v_full_j[:,j+1]
    return h_j,v_full_j,v_j

def back_substitution(h,g, m):
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
    m = A.shape[0]
    L = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1):
            L[i][j] = A[i][j]

    return L

def apply_preconditioner(M_inv, matrix, precond=None):
    if precond == 'Jacobi':
        #print('Using Jacobi preconditioning')
        return M_inv @ matrix
    elif precond == 'GS':
        #print('Using Gauss Seidel preconditioning')
        return forward_substitution(M_inv, matrix)
    else:
        #print('No preconditioning')
        return matrix

def apply_preconditioner_msr(M_inv, matrix, precond=None):
    if precond == 'Jacobi':
        return M_inv * matrix
    elif precond == 'GS':
        #print('Using Gauss Seidel preconditioning')
        return forward_substitution(M_inv, matrix)
    else:
        #print('No preconditioning')
        return matrix
    
def return_Minv(M, precond=None):
    if precond == 'Jacobi':
        #print('M_inv = Jacobi preconditioning')
        return np.linalg.inv(M)
    elif precond == 'GS':
        #print('M_inv = Gauss Seidel preconditioning')
        return lower_matrix(M)
    else:
        #print('M_inv = Identity matrix')
        return M
    
def return_Minv_msr(M, precond=None):
    if precond == 'Jacobi':
        n_d = len(M)
        Minv = np.zeros((n_d))
        for i in range(n_d):
            Minv[i] = 1.0/M[i]
    elif precond == 'GS':
        Minv = M
    else:
        Minv = np.identity(M.shape[0])
    
    return Minv

# [x] TODO: implemented GMRES without preconditioning (completed wp2)
def gmres_(A,b, x_0, M=None, maxiter=None, tol=1e-08, precond=None, file: IO = None):

    m = A.shape[0]
    if precond is None: M = np.identity(m)
    M_inv = return_Minv(M, precond)
    r_0 = b - matrix_vector_product(A,x_0)
    r_0 = apply_preconditioner(M_inv, r_0, precond)  #M_inv @ r_0   # Preconditioning
    # print(f"r_0 custom precond = {r_0}")
    # print(f"r_0 scipy precond = {np.linalg.inv(M) @ (b - matrix_vector_product(A,x_0))}")
    assert np.allclose(r_0, np.linalg.inv(M) @ (b - matrix_vector_product(A,x_0))), "Preconditioning is not correct"
    break_cond = tol
    v_norm = r_0/norm(r_0)   # r0/norm(r0) where r0 = np.array([1,2,3,4]) for convinience but should be r0 = b - Ax0
    h = np.zeros((m+1,m))  # hessenberg matrix
    v_full = np.zeros((m,m+1))  # orthonormal basis
    v_full[:, 0] = v_norm
    c, s = [],[]
    g = [norm(r_0)]  # g = norm(r0)
    if maxiter is None: maxiter = A.shape[0] 

    for j in range(maxiter):
        h,v_full,vp1 = get_krylov(A, v_full, j, h, M_inv, precond)
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
        g.append(-s[j]*g[j])   # needs exit condition
        g[j] = c[j]*g[j]
        rho = abs(g[j+1])
        rel_res = rho/abs(g[0])
        file.write(f"{j} , {rel_res}\n") 
        if rho < break_cond:
            break

    m_bar = min(m,j+1)
    y = back_substitution(h[:m_bar+1][:m_bar],g[:m_bar+1],m_bar)
    x = x_0 + matrix_vector_product(v_full[:,:m_bar],y)

    return h,v_full,x,rel_res

def gmres_msr(A_JM, A_VM,b, x_0, M=None, maxiter=None, tol=1e-08, precond=None,file: IO = None, symm=False, rho_0=1.0, Minv=None):

    m = b.shape[0]
    if precond is None: M = np.identity(m)
    M_inv = Minv #return_Minv_msr(M, precond)
    r_0 = b - MSR_matmul(A_JM, A_VM, x_0, symm=symm)      #matrix_vector_product(A,x_0)
    r_0 = apply_preconditioner_msr(M_inv, r_0, precond)  #M_inv @ r_0   # Preconditioning
    break_cond = tol
    v_norm = r_0/norm(r_0)   # r0/norm(r0) where r0 = np.array([1,2,3,4]) for convinience but should be r0 = b - Ax0
    h = np.zeros((m+1,m))  # hessenberg matrix
    v_full = np.zeros((m,m+1))  # orthonormal basis
    v_full[:, 0] = v_norm
    c, s = [],[]
    g = [norm(r_0)]  # g = norm(r0)
    if maxiter is None: maxiter = m 

    for j in range(maxiter):
        h,v_full,vp1 = get_krylov_msr(A_JM, A_VM,v_full, j, h, M_inv, precond, symm=symm)
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
        g.append(-s[j]*g[j])   # needs exit condition
        g[j] = c[j]*g[j]
        rho = abs(g[j+1])
        rel_res = rho/abs(norm(rho_0))
        file.write(f"{j} , {rho}\n")
        if rho < break_cond:
            break

    m_bar = min(m,j+1)
    y = back_substitution(h[:m_bar+1][:m_bar],g[:m_bar+1],m_bar)
    x = x_0 + matrix_vector_product(v_full[:,:m_bar],y)

    return x,rho

def restart_gmres(A,b,x_0,M=None,m=20,tol=1e-08,precond=None):
    r = b - matrix_vector_product(A,x_0)
    rho = norm(r)/norm(r)
    # open file here to write the residuals
    with open("gmres_residuals_right_" + str(m) +".txt", "w") as file:
        file.write("Iter , rho\n")
        x = x_0
        count = 0
        while rho > tol:
            h_g, v_g, x_m, rho = gmres_(A, b, x, M, m, tol, precond, file)
            x = x_m
            count +=1
            if count > 50:
                print("GMRES did not converge in 50 iterations")
                break

    return x_m

def restart_gmres_msr(A_JM, A_VM,b,x_0,M,m,tol,precond, symm=False, max_while_count=500):
    r = b - MSR_matmul(A_JM, A_VM, x_0, symm=symm)
    M_inv = return_Minv_msr(M, precond)
    r = apply_preconditioner_msr(M_inv, r, precond)
    rho = norm(r)/norm(r)
    # open file here to write the residuals
    with open("gmres_residuals_" + precond + str(m) +".txt", "w") as file:
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
            else:
                pass
                #print("Iteration set : ", count)

    #np.savetxt("ortho_kyrlov.txt", v_g)
    print("GMRES converged in {} iterations".format(count))
    print("Final residual: ", rho)
    print('Final relative residual: ', rho/norm(r))

    return x_m

def err_arnoldi(v_arnoldi, h_arnoldi, a_matrix):
    return la.norm(v_arnoldi[:,:-1].T @ a_matrix @ v_arnoldi[:, :-1] - h_arnoldi[:-1])/la.norm(a_matrix)


#[x] TODO: implemented conjugate gradient wp3
def conjugate_gradient(A_, b_, x0_cg, tol=1e-8, maxiter=10):
    r0 = b_ - A_ @ x0_cg
    p0 = r0
    xm = x0_cg
    rm = r0
    pm = p0
    for i in range(maxiter):  # Changed from size to maxiter
        Apm = A_ @ pm
        alpha_m = np.dot(rm,rm) / np.dot(Apm, pm)
        xmp1 = xm + alpha_m * pm
        rmp1 = rm - alpha_m * Apm
        beta_m = np.dot(rmp1, rmp1) / np.dot(rm, rm)
        pmp1 = rmp1 + beta_m * pm
        xm = xmp1
        rm = rmp1
        pm = pmp1
        print(f"iteration {i} x = {xmp1}")
        print(f"Conjugate gradient did not converge in {maxiter} iterations")
        if np.linalg.norm(rmp1) < tol:  # Check for convergence
            break

    return xmp1

def conjugate_gradient_msr(A_JM, A_VM, b_, x0_cg, tol=1e-8, maxiter=10, symm=True):
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

    return xmp1
