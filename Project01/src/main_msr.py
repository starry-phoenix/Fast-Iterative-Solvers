import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from workpackage import gmres_msr, conjugate_gradient_msr, get_M_msr

# Example MSR matrix and vector b (assuming these are defined in workpackage.py)
# For demonstration, let's assume msr_matrix is the MSR format matrix
# and msr_to_dense is a function that converts MSR format to a dense matrix
msr_matrix = {
    'values': [1,2,1,2,0,4,2,4],
    'index': [6,7,8,8,9,3,3,3]
}
b = np.array([1, 2, 3, 4])
X0 = np.zeros_like(b)

# get M matrix 
M = get_M_msr(msr_matrix['index'], msr_matrix['values'])

# Define tolerance and maximum iterations
tol = 1e-10
maxiter = 1000

# Solve using GMRES
x_gmres, gmres_convergence = gmres_msr(msr_matrix['index'], msr_matrix['values'], b, X0, tol, maxiter, M)

# Solve using Conjugate Gradient
x_cg, cg_convergence = conjugate_gradient_msr(msr_matrix['index'], msr_matrix['values'], b, X0, tol, maxiter, M)

# read data
data_residuals_gmres = pd.read_csv(gmres_convergence, delim_whitespace=True)
residuals_gmres = data_residuals_gmres['rho']

# read data
data_residuals_cg = pd.read_csv(cg_convergence, delim_whitespace=True)
residuals_cg = data_residuals_cg['rho']

# Plotting convergence history
plt.figure(figsize=(10, 5))
plt.plot(residuals_gmres, label='GMRES Convergence')
plt.plot(cg_convergence, label='CG Convergence')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Convergence History')
plt.legend()
plt.grid(True)
plt.show()