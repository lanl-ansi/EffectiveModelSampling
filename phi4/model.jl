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


function batch_gradient(batch_t, θ, ∇J)
    d = length(θ)
    acc = zeros(d)

    @threads for i in size(batch_t)[2]
        acc .+= ∇J(θ, batch_t[i,:])
    end

    return acc ./ length(batch_t)
end


function adamGD(obs, err, ∇J, θ, B; β1=0.9, β2=0.999, ε=1e-8, α=1e-3, tol=1e-4)
    println("starting Adam...")

    m = zeros(size(θ))
    v = zeros(size(θ))
    t = 0

    g_prev = fill(Inf, length(θ))
    θ_plus1 = Inf

    while true #norm(g_prev) > err
        batch_t = Matrix(obs[rand(1:10000, B), :])
        t += 1
        g_t = batch_gradient(batch_t, θ, ∇J) #parallelizeable

        norm_grad = norm(g_t)
        if t % 10 == 0
            println("T=$t | θ=$(round.(θ, digits=6)) | ‖∇J‖=$(round(norm_grad, digits=6))")
        end

        if norm_grad ≤ err
            break
        end

        g_prev .= g_t

        m = β1 .* m .+ (1 - β1) .* g_t
        v = β2 .* v .+ (1 - β2) .* (g_t .^ 2)
        m_hat = m ./ (1 .- β1^t)
        v_hat = v ./ (1 .- β2^t)

        θ_plus1 = θ .- α .* m_hat ./ (sqrt.(v_hat) .+ ε)

        if norm(θ_plus1 - θ) < tol
            break
        end

        if any(isinf, θ_plus1) || any(isnan, θ_plus1)
            α *= 0.1    # reduce stepsize
        else
            θ = θ_plus1
        end
    end
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

f(x, θ) = θ[1]*sum(x.^4) + θ[2]*sum(x.^2) + 2*θ[3]*sum([x[src(e)]*x[dst(e)] for e in edges(g)])
df_dxj(x, j, θ, Ax) =4*θ[1]*x[j]^3 + 2*θ[2]*x[j] + 2*θ[3]*Ax[j]
q(x, θ) = exp(-f(x, θ))

dJf_dθ1(θ, subsamples)=mean([sum([df_dxj(x, j, θ, Array(A*x))*(4*x[j]^3)-12*x[j]^2 for j in 1:D]) for x in eachcol(subsamples)])
dJf_dθ2(θ, subsamples)=mean([sum([df_dxj(x, j, θ, Array(A*x))*(2*x[j])-2 for j in 1:D]) for x in eachcol(subsamples)])
dJf_dθ3(θ, subsamples)=mean([sum([df_dxj(x, j, θ, Array(A*x))*(2*Array(A*x)[j]) for j in 1:D]) for x in eachcol(subsamples)])


∇J_poly(θ, subsamples) = [dJf_dθ1(θ, subsamples), dJf_dθ2(θ, subsamples), dJf_dθ3(θ, subsamples)]


thetaReal = [0.1, 0.3, 0.3] # two humps
f_obs(x) = f(x, thetaReal)
q_obs(x) = exp(-f_obs(x))

range = [-5, 5]
ϵ = 0.0001
# Generate x values for plotting
err = 1e-4
B=1000

println("re-learning thetas . . .")
startθ = [0.11, 0.33, 0.33]
scoreMatchθ = adamGD(Matrix(samples'), err, ∇J_poly, startθ, B)
#scoreMatchθ = thetaReal
println("real =", thetaReal)
println("scorematch theta est =", scoreMatchθ)
q_real(x) = exp(-f(x, thetaReal)) 
q_inf(x) = exp(-f(x, scoreMatchθ)) 

println("rejection sampling to obtain p(scorematch theta) from p_Gauss(gaussian est) samples")
#Nsamp = N#10000
p_gmm = MixtureModel(GMM(2, Matrix(samples'); method=:kmeans))
RSsamples, M, acc, rej=RejectionSampling.rejectionSampling(Nsamp, q_inf, p_gmm; checkM=false)
inferred = hcat(RSsamples)
samples_matrix_gmm = rand(p_gmm, Nsamp)   # 3 × 10000
samples_vecvec_gmm = [samples_matrix_gmm[:, i] for i in 1:size(samples_matrix_gmm, 2)]
df_gmmpdf  = safe_dataframe(samples_vecvec_gmm, D, "gmm.csv")

df_obs = DataFrame(transpose(samples), :auto)
fig1 = safe_pairplot(df_obs, "observed data (10k samples)")
save("pairplot_true.png", fig1)

df_inf = safe_dataframe(RSsamples, D, "inferred.csv")
fig2 = safe_pairplot(df_inf, "inferred data (10k samples)")
save("pairplot_inferred.png", fig2)

fig3 = safe_pairplot(df_gmmpdf, "effective gmm pdf (10k samples)\nM=$M")
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
rows, cols = 3, 1
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

