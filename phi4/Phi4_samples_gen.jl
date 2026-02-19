using Graphs
using GraphPlot 
using LinearAlgebra
using DataFrames, CSV
using PairPlots, Plots
using Images, ImageMagick, ImageTransformations
using CairoMakie


function score_func(x,A,p,g)
    grad_log_density = -4*p[1]*x.^3 - 2*(p[2]*I + p[3]*A)*x
    return grad_log_density
end

function Langevin_Dynamics(x0,A,p,T, g)
    dt = 0.001
    N = size(A,1)

    T_burn = 100000
    x_burn = zeros(N,T_burn)
    x = x0
    for k=1:T_burn
        x = x + dt*score_func(x,A,p,g) + sqrt(2*dt)*randn(N)
        x_burn[:,k] = x
    end

    x_traj = zeros(N,T)
    x = x_burn[:,end]
    for k=1:T
        x = x + dt*score_func(x,A,p,g) + sqrt(2*dt)*randn(N)
        x_traj[:,k] = x
    end

    return x_burn, x_traj
end

################################# Sample Generation ################################################
# create graph/adj_matrix on lattice  
g = Graphs.SimpleGraphs.grid((4,4))
gplot(g, nodelabel=1:nv(g))
A = adjacency_matrix(g)

error_compress = zeros(5,3)
error_MCMC = zeros(5,3)

p = [0.1, 0.3, 0.3]

# Sampling with Langevin_Dynamics
N = 16
n_samples = 10000
skipstep = 100000
(x_burn, x_traj) = Langevin_Dynamics(zeros(N,),A,p,skipstep*n_samples, g)
S1 = x_traj[:,1:skipstep:end]
df = DataFrame(S1, :auto)  # transpose so each row is a sample (optional)

# Save to CSV
CSV.write("data1.csv", df)
fig= pairplot(DataFrame(transpose(S1), :auto))
save("pairplot_true1.png", fig)
