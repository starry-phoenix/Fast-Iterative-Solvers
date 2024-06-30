using CSV, DataFrames

function dot(A,B)
    sum = 0
    for i in eachindex(A)
        sum += A[i]*B[i]
    end
    return sum
end

function norm(A)
    sum = 0
    for i in eachindex(A)
        sum += A[i]^2
    end
    return sqrt(abs(sum))
end

function MSR_matmul(JM, VM, X, symm=false)
    n_d = findfirst(VM .== 0.0)-1    # number of diagonal elements
    V_d = VM[1:n_d]   # until the pointer = diagonal elements
    V_nd = VM[n_d+2:end]  # after the pointer = non-diagonal elements
    IA_M = JM[1:n_d+1]  # CSR row pointer for non-diagonal elements
    J_M = JM[n_d+2:end]  # CSR column index for non-diagonal elements
    YM = zeros(n_d)

    for i in 1:n_d
        YM[i] = V_d[i] * X[i]
    end

    n = length(IA_M) - 1
    IA_M = IA_M .- (n_d + 1)
    for i in 1:n   # row index
        i1 = IA_M[i]
        i2 = IA_M[i+1] - 1
        for j in i1:i2
            YM[i] += V_nd[j] * X[J_M[j]]
            if symm && (J_M[j]) != i
                YM[J_M[j]] += V_nd[j] * X[i]   # Update the YM array by adding the product of V_nd[j] and X[i] to YM[J_M[j] - 1]
            end
        end
    end
    return YM
end

function conjugate_gradient_msr(A_JM::Vector{Int64}, A_VM::Vector{Float64}, b_::Vector{Float64}, x0_cg::Vector{Float64}, tol::Float64=1e-08, maxiter::Int=10, symm::Bool=false)
    r0 = b_ - MSR_matmul(A_JM, A_VM, x0_cg, symm)
    p0 = r0
    xm = x0_cg
    rm = r0
    pm = p0
    rel_res = 1.0
    x = ones(length(x0_cg))
    r0_res = norm(b_)
    i = 0
    open("cg_residuals.txt", "w") do file
        write(file, "Iter,res,rho,e_Anorm\n")    #,e_anorm\n")
        while rel_res> tol && i<maxiter  # Changed from size to maxiter
            Apm = MSR_matmul(A_JM, A_VM, pm, symm)
            alpha_m = dot(rm, rm) / dot(Apm, pm)
            xmp1 = xm + alpha_m * pm
            rmp1 = rm - alpha_m * Apm
            beta_m = dot(rmp1, rmp1) / dot(rm, rm)
            pmp1 = rmp1 + beta_m * pm
            xm = xmp1
            rm = rmp1
            pm = pmp1
            res = norm(rmp1)
            rel_res = res / r0_res
            err_k = norm(xmp1 - x)
            # Ae = MSR_matmul(A_JM, A_VM, err_k, symm)
            # e_Anorm = sqrt(abs(dot(Ae, err_k)))
            write(file, "$i,$res,$rel_res,$err_k\n")    #, $e_Anorm\n")
            i += 1
        end
    end
    if i == maxiter
        println("\nConjugate gradient did not converge in $maxiter iterations")
    else
        println("\nConjugate gradient converged in $i iterations")
    end
    return xm
end

# r = [1.0,4.0,5.0,6.0]
# # X = [1.0,1.0,1.0,1.0]

# J_Msr = [6,7,8,8,9,3,3,3]   # Matrix index
# V_Msr = [1.0,2.0,1.0,2.0,0.0,4.0,2.0,4.0]  # Matrix values
# tol = 1e-05
# maxiter = 10

# # x_julia_msr = MSR_matmul(J_Msr, V_Msr, X, false)


# #x_cg_msr = conjugate_gradient_msr(J_Msr, V_Msr, r, X, tol, maxiter, false)

# # Assuming spaces are used to separate fields in your FWF file

# MAIN CODE
data_cg = CSV.File("cg_test_msr.txt", delim=' ', ignorerepeated=true,silencewarnings=true) |> DataFrame
# data_cg = CSV.read("cg_test_msr.txt", DataFrame, delim=' ', ignorerepeated=true, silencewarnings=true)
Vm_gmres = Vector(data_cg[2:end,2])
Jm_gmres = Vector(data_cg[2:end,1])
print("Vm shape: ", length(Vm_gmres))
print("\nJm shape: ", length(Jm_gmres))

symm = true
x_test = ones((length(Vm_gmres)))
r = MSR_matmul(Jm_gmres,Vm_gmres, x_test, symm)
len_gmres = length(r)
tol = 1e-08
X = zeros(len_gmres)
maxiter=1000

execution_time = @elapsed x_cg_msr = conjugate_gradient_msr(Jm_gmres, Vm_gmres, r, X, tol,100000,symm)
