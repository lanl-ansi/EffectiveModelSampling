# =======================================================
# main.jl
# julia -p 20 main.jl
# =======================================================

using Distributions, Plots, Base.Threads
using Random, CSV, DataFrames
using Graphs
using GraphPlot
using LinearAlgebra
using PairPlots
using CairoMakie
using GaussianMixtures
using Images, ImageMagick, ImageTransformations
using Distributions, Plots, Base.Threads, Random, LinearAlgebra, DataFrames, CSV, GaussianMixtures
include("../IMPORTABLES/MultiDNomial.jl")
using .MultiDNomial
include("../IMPORTABLES/RejectionSampling.jl")
using .RejectionSampling
include("../IMPORTABLES/ADAM_ScoreMatching.jl")
using .ADAM_ScoreMatching
include("../IMPORTABLES/Tools.jl")
using .Tools
Random.seed!(1234)

# ---------------------------
# Discretize a PDF and sample histogram
# ---------------------------
function discretizePDF(p_func, x_vals)
    ϵ = x_vals[2]-x_vals[1]
    midpoints_x = [(x_vals[i]+x_vals[i+1])/2 for i in 1:length(x_vals)-1]
    midpoints = [(x,y) for x in midpoints_x, y in midpoints_x]
    pdf_vals = [p_func(xy) for xy in midpoints]
    Z = sum(pdf_vals)*ϵ*ϵ
    pdf_vals_norm = pdf_vals ./ Z
    return midpoints, pdf_vals_norm, Z
end

function sampleHist(midpts, pdf_vals, x_vals, N)
    ϵ = x_vals[2]-x_vals[1]
    cdf = cumsum(vec(pdf_vals))*ϵ*ϵ
    cdf ./= cdf[end]
    u_samples = rand(N)
    #idx = [findfirst(x -> x ≥ u, cdf) for u in u_samples]
    idx = Vector{Int}(undef, N)
       @threads for i in 1:N
        idx[i] = findfirst(x -> x ≥ u_samples[i], cdf)
    end
    flat_midpts = vec(midpts)
    return flat_midpts[idx]
end

# ---------------------------
# Example: 1D polynomial
# ---------------------------
thetaReal =  [0.1, -0.0747442173335889, 0.12353334391668362, -0.00022082666146557226, 0.005843978159800084, 0.1619970188721066, -0.6790678970311284, 0.008353043177697873, 0.0076338132086725225, 0.20888153384651223, -0.004930100636868771, 0.0016801451426953252, 0.0030305984832797806, 0.0033879050992132935, -7.872011403848199e-5]
poly_obs = PolynomialModel(2, 4)
poly_obs.θ .= thetaReal 

# Define f_obs and q_obs via module functions
f_obs(x) = poly_obs(x)   # polynomial evaluation from your module
q_obs(x) = exp(-f_obs(x))

#range = -10:0.0001:10
range = -10:0.001:10
midpts, pdf_vals, Z = discretizePDF(q_obs, range)

println("Sampling from histogram ...")
N = 10^5
L = 4
D = 2

samples_ = sampleHist(midpts, pdf_vals, range, N)
samples_mat = [s[j] for s in samples_, j in 1:2]
samples = DataFrame(samples_mat, :auto)
#samples =  CSV.read("2Dsamples.csv", DataFrame)
samples_mat = Matrix(samples)

fig1 = safe_pairplot(samples, "observed data")
save("pairplot_true.png", fig1)
# ---------------------------
# Train polynomial via ADAM_SM
# ---------------------------
poly_inf = ADAM_ScoreMatching.MultiDNomial.PolynomialModel(2, 4)  # 1D, degree 4 polynomial
println("Training PolynomialModel with ADAM_SM ...")
samples_vec = [Vector(row) for row in eachrow(samples)]
adam_score_matching!(poly_inf, samples_vec; η=0.0001, tol_loss=1e-6)
println("Real θ = ", thetaReal)
println("Estimated θ (ADAM) = ", poly_inf.θ)

f_inf(x) = poly_inf(x)
q_inf(x) = exp(-f_inf(x))

# ---------------------------
# Rejection sampling against Gaussian approx
# ---------------------------
println("Fitting Gaussian approx via mean/std of samples ...")
p_gmm = MixtureModel(GMM(4, samples_mat; method=:kmeans))

println("Running rejection sampling ...")
Nsamp = N

RSsamples, M, accept, reject, mean_acc = RejectionSampling.rejectionSampling(Nsamp, q_inf, p_gmm)

# ---------------------------
# Plotting
# ---------------------------
inferred = hcat(RSsamples)
samples_matrix_gmm = rand(p_gmm, Nsamp)   # 3 × 10000
samples_vecvec_gmm = [samples_matrix_gmm[:, i] for i in 1:size(samples_matrix_gmm, 2)]
df_gmmpdf  = safe_dataframe(samples_vecvec_gmm, D, "gmm.csv")

fig1 = safe_pairplot(samples, "observed data (10k samples)")
save("pairplot_true.png", fig1)

df_inf = safe_dataframe(RSsamples, D, "inferred.csv")
fig2 = safe_pairplot(df_inf, "inferred data (10k samples)")
save("pairplot_inferred.png", fig2)

fig3 = safe_pairplot(df_gmmpdf, "effective gmm pdf (10k samples)")
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
#rows, cols = 3, 1
rows, cols = 1, 3
grid = reshape(imgs_rotated, cols, rows)'  # fill row-wise

# Horizontally concatenate each row
row_imgs = [hcat(grid[i, :]...) for i in 1:rows]

# Vertically concatenate rows
final_img = vcat(row_imgs...)

# Save the combined image
save("toy_2D.png", final_img)
println("mean acceptance=", mean_acc)
# uncomment if want moments
#gmm, inf, obs = getFiles("gmm.csv", "inferred.csv", "data1.csv")
#allMoments(gmm, inf, Matrix(obs'), outname="moments_summary11.txt")
println("finished.")
