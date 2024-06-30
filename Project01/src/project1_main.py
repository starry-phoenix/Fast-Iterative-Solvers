# Test data
import numpy as np
from workpackage import restart_gmres_msr, MSR_matmul, get_M_msr, conjugate_gradient_msr
import pandas as pd
import time
import matplotlib.pyplot as plt

# read data
data_cg = pd.read_fwf(r"cg_test_msr.txt")     #pd.read_fwf(r"gmres_test_msr.txt")
Vm_gmres = np.array(data_cg['Unnamed: 2'][1:])
Jm_gmres = np.array(data_cg['Unnamed: 1'][1:])
print("Vm shape: ", Vm_gmres.shape)
print("Jm shape: ", Jm_gmres.shape)
x_test = np.ones((Vm_gmres.shape[0]))
len_gmres = len(MSR_matmul(VM=Vm_gmres, JM=Jm_gmres, X=x_test, symm=True))

# set parameters
tol = 1e-04
restart_m = 600   # restart parameter
r = MSR_matmul(JM=Jm_gmres, VM=Vm_gmres,X=x_test, symm=True)   # = b
X = np.zeros((len_gmres))    # = x0

# Preconditioner
# Jacobi preconditioner
precond_ = 'Jacobi'
M_precond_msr = get_M_msr(JM=Jm_gmres,VM=Vm_gmres,symm=False, precond=precond_)
# Gauss Seidel preconditioner
precond_ = 'GS'
M_precond_gauss_msr = get_M_msr(JM=Jm_gmres,VM=Vm_gmres,symm=False, precond=precond_)

# GMRES---------------------
start = time.time()
y_gmres = restart_gmres_msr(Jm_gmres, Vm_gmres, r, X, M=None, m=restart_m, tol=tol, precond=None, symm=False, max_while_count=100)
end = time.time()

start = time.time()
y_gmres_precond = restart_gmres_msr(Jm_gmres, Vm_gmres, r, X, M=M_precond_gauss_msr, m=restart_m, tol=tol, precond=precond_, symm=False, max_while_count=50)
end = time.time()

print("GMRES time: ", end-start)

# CG-----------------------
start = time.time()
x_cg_msr = conjugate_gradient_msr(Jm_gmres, Vm_gmres, r, X, symm=True, tol=tol, maxiter=100000)
end = time.time()
print("CG time: ", end-start)


# Read residual data
data_residuals = pd.read_csv("gmres_residuals_"+ str(restart_m) + ".txt", delim_whitespace=True)
residuals = data_residuals['rho']

# Plot residual
plt.grid()
plt.plot(residuals)
plt.xlabel('Iteration')
plt.ylabel('Relative Residual')
plt.yscale('log')
plt.title('GMRES Residual Plot')
plt.show()

data_residuals = pd.read_csv("gmres_residuals_GS"+ str(restart_m) + ".txt", delim_whitespace=True)
residuals = data_residuals['rho']

# Plot residual
plt.grid()
plt.plot(residuals)
plt.xlabel('Iteration')
plt.ylabel('Relative Residual')
plt.yscale('log')
plt.title('GMRES Residual Plot')
plt.show()

#plot orthoganility
krylov_vec_dot = [np.dot(np.array(krylov_vec[0]), np.array(krylov_vec[i])) for i in range(1, len(krylov_vec))]

plt.grid()
plt.plot(krylov_vec_dot)
plt.xlabel('Iteration')
plt.ylabel('Orthogonality')
plt.yscale('log')
plt.title('GMRES Orthogonality Plot')
plt.show()

krylov_vec = pd.read_csv("ortho_kyrlov.txt", header=None, delim_whitespace=True, dtype=float)
krylov_vec_pd = pd.DataFrame(krylov_vec)

krylov_vec_rows = krylov_vec_pd.values.tolist()
print(krylov_vec_rows)