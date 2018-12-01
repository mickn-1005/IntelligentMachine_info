# Boltzmann Machineを実装してみた
# 実行環境　Julia1.0.0

using Plots

const theta = 0.0
const imgdim = 5
const itr = 10
# Boltzmann temperature params
temp = 1000

# Energy function == lyapnov from Ising model
lyapnov(w::AbstractMatrix,xr::AbstractVector) = xr' * w * xr + sum(theta * xr)
# Activate function == Temperature
function activation_temp(dE::AbstractFloat)
    prob = 1 / (1 + exp(-dE/temp))     # probability of del energy
    a = rand(1)
    if a > prob
        return 1
    else
        return 0
    end
end
