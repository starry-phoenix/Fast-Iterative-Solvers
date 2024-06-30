import numpy as np
from scipy.sparse import csr_matrix

IA = [1,3,5,6,8]   # row information pointer
J = [1,3,2,3,3,3,4]  # column information
V = [1,4,2,2,1,4,2]
X = [4,3,6,120]
Y = [0,0,0,0]
# column index > row index => upper triangular matrix
# column index < row index => lower triangular matrix
def CSR(IA,J,V,X):
    n = len(IA)-1
    for i in range(n):   # i = row index 
        i1 = IA[i]-1
        i2 = IA[i+1]-1
        sum = 0
        for j in range(i1,i2):
            sum += V[j]*X[J[j]-1]    # J[j-1] = column index 
        Y[i] = sum

    return Y

#Test with scipy
def csr_test():
    IA = [1-1,3-1,5-1,6-1,8-1]
    J = [1-1,3-1,2-1,3-1,3-1,3-1,4-1]
    V = [1,4,2,2,1,4,2]
    csrm = csr_matrix((V,J,IA), shape=(4,4)).toarray()
    #csrm = csr_matrix((V,J,IA))
    #print("CSR scipy is ".format(csr_matrix((V,J,IA), shape=(4,4))*X))

    return csrm

IA_csc = [1,2,3,7,8]  # column pointer
J_csc = [1,2,1,2,3,4,4]  # index information rows
V_csc = [1,2,4,2,1,4,2]
Y_csc = [0,0,0,0]

def CSC(IA,J,V,X):
    n = len(IA)-1
    for i in range(n):
        i1 = IA[i]-1
        i2 = IA[i+1]-1
        for j in range(i1,i2):
            Y_csc[J[j]-1] += V[j]*X[i]

    return Y_csc

csc_m = CSC(IA_csc,J_csc,V_csc,X)

J_M = np.array([6,7,8,8,9,3,3,3])   # Matrix index
VM = np.array([1,2,1,2,0,4,2,4])   # Matrix values

def MSR(JM,VM,X, symm = False):
    n_d = np.where(VM==0)[0][0]  # number of diagonal elements
    V_d = VM[:n_d]   # until the pointer = diagonal elements
    V_nd = VM[n_d+1:]  # after the pointer = non-diagonal elements
    n_nd = V_nd.shape[0]
    IA_M = JM[:n_d+1]  # CSR row pointer for non-diagonal elements
    J_M = JM[n_d+1:]  # CSR column index for non-diagonal elements
    YM = np.array([0,0,0,0])
    for i in range(n_d):
        YM[i] = V_d[i]*X[i]

    n = IA_M.shape[0]-1
    IA_M = IA_M - (n_d+1)
    Upper_mat = np.zeros((n,n))
    for i in range(n):   # row index
        i1 = IA_M[i]-1
        i2 = IA_M[i+1]-1
        for j in range(i1,i2):
            YM[i] += V_nd[j]*X[J_M[j]-1]
            if symm and (J_M[j]-1) != i:
                YM[J_M[j]-1] += V_nd[j]*X[i]
            else:
                pass    # J_M[j-1] = column index

    return YM

X = [1,1,1,1]
msr = MSR(J_M,VM,X)
csr_py = csr_test() @ X