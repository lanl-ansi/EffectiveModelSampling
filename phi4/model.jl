using Distributions, Plots, Base.Threads
using Random, CSV, DataFrames
using Graphs
using GraphPlot
using LinearAlgebra
using PairPlots
using CairoMakie
using GaussianMixtures
using Images, ImageMagick, ImageTransformations
include("../IMPORTABLES/RejectionSampling.jl")
using .RejectionSampling
include("../IMPORTABLES/Moments.jl")
using .Moments
include("../IMPORTABLES/ImportanceSampling.jl")
using .ImportanceSampling
include("../IMPORTABLES/Tools.jl")
using .Tools
using Printf

# Helper: safely create DataFrame from samples
# Helper: safely create DataFrame from samples
function safe_dataframe(samples::Vector{Vector{Float64}}, D::Int, csv_name::String)
    if isempty(samples)
        df = DataFrame()
        for i in 1:D
            df[!, Symbol("x$i")] = Float64[]
        end
        CSV.write(csv_name, df)
        return df
    else
        mat = Matrix(reduce(hcat, samples)')  # convert Adjoint to Matrix
        df = DataFrame(mat, :auto)            # automatically name columns x1, x2, ...
        CSV.write(csv_name, df)
        return df
    end
end


# Helper: safely create pairplot from DataFrame
function safe_pairplot(df::DataFrame, title::String)
    if isempty(df)
        # blank Figure
        fig = Figure(resolution=(400,400))
        ax = Axis(fig[1,1])
    else
        fig= pairplot(df)
        Label(fig[0,:], title)
    end
    return fig
end

# Helper: safely create pairplot from DataFrame
function safe_pairplot(df::DataFrame, title::String)
    if isempty(df)
        # blank Figure
        fig = Figure(resolution=(400,400))
        ax = Axis(fig[1,1])
    else
        fig = pairplot(df)
        Label(fig[0, :], title)
    end
    return fig
end


function adamGD(obs, err, ∇J, θ, B; β1=0.9, β2=0.999, ε=1e-8,
                α=1e-4, decay_steps=5_000.0,
                maxiter=100_000, check_every=100,
                coercivity_floor=1e-8)
    println("starting Adam...")

    coercivity_floor > 0 || error("coercivity_floor must be positive")
    B > 0 || error("B must be positive")
    α > 0 || error("α must be positive")
    decay_steps > 0 || error("decay_steps must be positive")
    maxiter > 0 || error("maxiter must be positive")
    check_every > 0 || error("check_every must be positive")
    θ = Float64.(θ)
    θ[1] = max(θ[1], coercivity_floor)

    m = zeros(size(θ))
    v = zeros(size(θ))
    t = 0

    N = size(obs, 2)  # obs is D × N: one observation per column
    converged = false

    while t < maxiter
        indices = rand(1:N, min(B, N))
        batch_t = Matrix(obs[:, indices])
        t += 1
        g_t = ∇J(θ, batch_t)

        all(isfinite, g_t) || error("non-finite score-matching gradient")

        m = β1 .* m .+ (1 - β1) .* g_t
        v = β2 .* v .+ (1 - β2) .* (g_t .^ 2)
        m_hat = m ./ (1 .- β1^t)
        v_hat = v ./ (1 .- β2^t)

        # Reduce the Adam step gradually.  The initial step is α and the
        # effective step becomes smaller as the iterates approach a minimum.
        α_t = α / sqrt(1 + t / decay_steps)
        θ_plus1 = θ .- α_t .* m_hat ./ (sqrt.(v_hat) .+ ε)

        if !all(isfinite, θ_plus1)
            α *= 0.1    # reduce stepsize
            continue
        end

        θ_plus1[1] = max(θ_plus1[1], coercivity_floor)
        θ = θ_plus1

        # Do not stop using one noisy minibatch gradient. Check the complete
        # empirical objective periodically instead.
        if t % check_every == 0
            full_gradient = ∇J(θ, obs)
            if θ[1] == coercivity_floor && full_gradient[1] > 0
                full_gradient[1] = 0.0  # projected gradient at the boundary
            end
            full_grad_norm = norm(full_gradient)
            println("T=$t | θ=$(round.(θ, digits=6)) | ‖∇J‖=$(round(full_grad_norm, digits=6))")
            if full_grad_norm ≤ err
                converged = true
                break
            end
        end
    end
    !converged && println("Warning: Adam stopped without meeting err")
    println("done")
    return θ
end



Nsamp = 10000 #10k
L = 4
D = 16
df = CSV.read("data1.csv", DataFrame)
samples = Matrix(df)

g = Graphs.grid((4, 4))
gplot(g, nodelabel=1:nv(g))  # optional: visualize the lattice
A = adjacency_matrix(g)      # A is now the adjacency matrix (sparse)

# Convention preserved from the original code: each undirected edge is listed
# once, but theta[3] multiplies twice the edge sum. If the data generator used
# this convention, the edge-once coefficient in the paper is gamma=2*theta[3].
f(x, θ) = θ[1]*sum(x.^4) + θ[2]*sum(x.^2) + 2*θ[3]*sum(x[src(e)]*x[dst(e)] for e in edges(g))
df_dxj(x, j, θ, Ax) =4*θ[1]*x[j]^3 + 2*θ[2]*x[j] + 2*θ[3]*Ax[j]
q(x, θ) = exp(-f(x, θ))

function ∇J_poly(θ, subsamples)
    n = size(subsamples, 2)
    gradient = zeros(3)

    for x in eachcol(subsamples)
        Ax = A * x
        grad_f = 4*θ[1] .* x.^3 + 2*θ[2] .* x + 2*θ[3] .* Ax

        # J(theta) = mean[ ||grad f||^2/2 - Delta f ].
        # Delta f = 12*theta[1]*sum(x.^2) + 2*theta[2]*D;
        # the interaction has zero Laplacian because diag(A) = 0.
        gradient[1] += dot(grad_f, 4 .* x.^3) - 12*sum(x.^2)
        gradient[2] += dot(grad_f, 2 .* x) - 2*length(x)
        gradient[3] += dot(grad_f, 2 .* Ax)
    end

    return gradient ./ n
end


thetaReal = [0.1, 0.3, 0.3] # two humps
f_obs(x) = f(x, thetaReal)
q_obs(x) = exp(-f_obs(x))

range = [-5, 5]
ϵ = 0.0001
# Generate x values for plotting
err = 1e-4
B=1000

println("re-learning thetas . . .")
startθ = rand(3)#[0.11, 0.33, 0.33]
scoreMatchθ = adamGD(samples, err, ∇J_poly, startθ, B)
println("real =", thetaReal)
println("scorematch theta est =", scoreMatchθ)
q_real(x) = exp(-f(x, thetaReal))
q_inf(x) = exp(-f(x, scoreMatchθ))

println("rejection sampling to obtain p(scorematch theta) from p_Gauss(gaussian est) samples")
p_gmm = MixtureModel(GMM(4, Matrix(samples'); method=:kmeans))
RSsamples, M, acc, rej, mean_acc = RejectionSampling.rejectionSampling(Nsamp, q_inf, p_gmm; checkM=false)
inferred = hcat(RSsamples)
samples_matrix_gmm = rand(p_gmm, Nsamp)  
samples_vecvec_gmm = [samples_matrix_gmm[:, i] for i in 1:size(samples_matrix_gmm, 2)]
df_gmmpdf  = safe_dataframe(samples_vecvec_gmm, D, "gmm.csv")

df_obs = DataFrame(transpose(samples), :auto)
fig1 = safe_pairplot(df_obs, "observation")#"observed data (10k samples)")
save("pairplot_true.png", fig1)

df_inf = safe_dataframe(RSsamples, D, "inferred.csv")
fig2 = safe_pairplot(df_inf, "inferred")#"inferred data (10k samples)")
save("pairplot_inferred.png", fig2)

fig3 = safe_pairplot(df_gmmpdf, "gmm")#"effective gmm pdf (10k samples)\nM=$M")
save("pairplot_gmm.png", fig3)

png_files = [
    "pairplot_true.png",
    "pairplot_inferred.png",
    "pairplot_gmm.png"
]

# Load images
imgs = [load(file) for file in png_files]

# Resize all images to the size of the first image (optional but ensures alignment)
target_h, target_w = size(imgs[1])
imgs_resized = [imresize(img, (target_h, target_w)) for img in imgs]

# Flip each image vertically to fix mirrored text
imgs_corrected = [reverse(img, dims=1) for img in imgs_resized]

# Rotate each image 90° clockwise three times (equivalent to 90° counterclockwise)
imgs_rotated = [rotr90(img) for img in imgs_corrected]

# Arrange images in 3 rows x 2 columns
rows, cols = 1, 3
grid = reshape(imgs_rotated, cols, rows)'  # fill row-wise

# Horizontally concatenate each row
row_imgs = [hcat(grid[i, :]...) for i in 1:rows]

# Vertically concatenate rows
final_img = vcat(row_imgs...)

# Save the combined image
save("phi4.png", final_img)

gmm, inf, obs = getFiles("gmm.csv", "inferred.csv", "data1.csv")
allMoments(gmm, inf, Matrix(obs'), outname="moments_summary11.txt")
println("finished.")


 
obsM = Matrix(obs')        # N×D, same convention as the allMoments(...) call (obs from getFiles is D×N)
 
# IS moments of the learned model p ∝ exp(-f(x, θ̂)).
# Target FIRST (unnormalized q_inf), proposal SECOND (p_gmm).
# Z_inf is not available in 16-D (no grid sum as in toy_1D), so the weights are
# self-normalized, which is the default of this function.
#isres = importanceSamplingTensorMoments(Nsamp, q_inf, p_gmm)
#allMomentsIS(isres, Matrix(obs'); outname="moments_summary11_IS.txt")
 
#moment = (firstMoment, secondMoment, thirdMoment, fourthMoment)   # from Moments.jl
#is_est = (isres.m1, isres.m2, isres.m3, isres.m4)
 
#println("\nRMS error of the E[X^⊗k] tensors vs. the observed data (same metric as allMoments)")
#@printf("%-8s  %14s  %14s  %14s\n", "moment", "gmm", "RS (inferred)", "IS (inferred)")
#for k in 1:4
#    ref = moment[k](obsM)
#    @printf("%-8d  %14.4e  %14.4e  %14.4e\n", k,
#            Moments.getErr(moment[k](gmm), ref),
#            Moments.getErr(moment[k](inf), ref),
#            Moments.getErr(is_est[k], ref))
#end
#@printf("IS effective sample size: %.0f of %d\n", isres.ess, Nsamp)
