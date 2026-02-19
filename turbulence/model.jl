# =======================================================
# main.jl
# julia -p 20 main.jl
# =======================================================

using Distributions, Plots, Base.Threads, MultivariateStats
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
include("../IMPORTABLES/Moments.jl")
using .Moments
include("utils.jl")
using .Utils

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
    end
    return fig
end

L = 6
D = 3
data_og = Utils.load_data()
datafull = Utils.coarse_grain(data_og)
data = datafull[:, :, :, [1,8,9]]
N1, N2, N3, Dim = size(data)
samples_mat = reshape(data, N1*N2*N3, Dim)
samples = DataFrame(samples_mat, :auto)
N = N1*N2*N3
#sample_indices = sample(1:nrow(samples_og), N; replace = false)
#samples = samples_og[sample_indices, :]
#samples_mat = Matrix(samples)
#bork
println("size=", size(samples_mat))
CSV.write("observation.csv", samples)

fig1 = safe_pairplot(samples, "observed data")
save("pairplot_true.png", fig1)
# ---------------------------
# Train polynomial via ADAM_SM
# ---------------------------
poly_inf = ADAM_ScoreMatching.MultiDNomial.PolynomialModel(D, L)  # 1D, degree 4 polynomial
println("Degree = ", poly_inf.L, "  Dimension=", poly_inf.D)
println("Training PolynomialModel with ADAM_SM ...")
samples_vec = [Vector(row) for row in eachrow(samples_mat)]
adam_score_matching!(poly_inf, samples_vec; η=0.001, tol_loss=1e-3)
println("Estimated θ (ADAM) = ", poly_inf.θ)

f_inf(x) = poly_inf(x)
q_inf(x) = exp(-f_inf(x))

# ---------------------------
# Rejection sampling against Gaussian approx
# ---------------------------
println("Fitting Gaussian approx via mean/std of samples ...")
p_gmm = MixtureModel(GMM(1, samples_mat; method=:kmeans))

println("Running rejection sampling ...")
Nsamp = N

RSsamples, M, accept, reject = RejectionSampling.rejectionSampling(Nsamp, q_inf, p_gmm, checkM=false)

#CSV.write("POLYNOMIAL1D_SAMPLES_inf.csv", df_inf)

# ---------------------------
# Plotting
# ---------------------------
#RSsamples, M, acc, rej=RejectionSampling.rejectionSampling(Nsamp, q_inf, p_gmm; checkM=false)
inferred = hcat(RSsamples)

samples_matrix_gmm = rand(p_gmm, Nsamp)   # 3 × 10000
samples_vecvec_gmm = [samples_matrix_gmm[:, i] for i in 1:size(samples_matrix_gmm, 2)]
df_gmmpdf  = safe_dataframe(samples_vecvec_gmm, D, "gmm.csv")

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
save("turbulence.png", final_img)
gmm, inf, obs = getFiles("gmm.csv", "inferred.csv", "observation.csv")
allMoments(gmm, inf, obs, outname="moments_summary.txt")
println("finished.")
