# =======================================================
# main.jl
# julia -p 20 main.jl
# =======================================================
using Distributions, Plots, Base.Threads, Random, LinearAlgebra, DataFrames, CSV, GaussianMixtures
include("../IMPORTABLES/MultiDNomial.jl")
using .MultiDNomial
include("../IMPORTABLES/RejectionSampling.jl")
using .RejectionSampling
include("../IMPORTABLES/ADAM_ScoreMatching.jl")
using .ADAM_ScoreMatching
Random.seed!(1234)
# ---------------------------
# Discretize a PDF and sample histogram
# ---------------------------
function discretizePDF(p_func, x_vals)
    ϵ = x_vals[2]-x_vals[1]
    midpoints = [(x_vals[i]+x_vals[i+1])/2 for i in 1:length(x_vals)-1]
    pdf_vals = [p_func(x) for x in midpoints]
    Z = sum(pdf_vals)*ϵ
    pdf_vals_norm = pdf_vals ./ Z
    return midpoints, pdf_vals_norm, Z
end

function sampleHist(midpts, pdf_vals, x_vals, N)
    ϵ = x_vals[2]-x_vals[1]
    cdf = cumsum(pdf_vals).*ϵ
    cdf ./= cdf[end]
    u_samples = rand(N)
    idx = [findfirst(x -> x ≥ u, cdf) for u in u_samples]
    return midpts[idx]
end


# ---------------------------
# Example: 1D polynomial
# ---------------------------
thetaReal = [3.7, -2.0, -8.33, 5.0]   # set true coefficients
poly_obs = PolynomialModel(1, 4)
poly_obs.θ .= [0, 3.7, -2.0, -8.33, 5.0]   # set true coefficients

# Define f_obs and q_obs via module functions
f_obs(x) = poly_obs([x])   # polynomial evaluation from your module
q_obs(x) = exp(-f_obs(x))

range = -5:0.0001:5
midpts, pdf_vals, Z = discretizePDF(q_obs, range)

println("Sampling from histogram ...")
N = 10^5
samples = sampleHist(midpts, pdf_vals, range, N)

# ---------------------------
# Train polynomial via ADAM_SM
# ---------------------------
poly_inf = ADAM_ScoreMatching.MultiDNomial.PolynomialModel(1, 4)  # 1D, degree 4 polynomial
println("Training PolynomialModel with ADAM_SM ...")
samples_vec = [[x] for x in samples]  # convert Float64 → Vector{Float64}
adam_score_matching!(poly_inf, samples_vec; η=0.0001, tol_loss=1e-4)
#poly_inf.θ = [3.444049774147762, -2.0848089876659675, -8.057527470095437, 4.900050617908192]

println("Real θ = ", thetaReal)
println("Estimated θ (ADAM) = ", poly_inf.θ)

f_inf(x) = poly_inf([x])
q_inf(x) = exp(-f_inf(x))

# ---------------------------
# Rejection sampling against Gaussian approx
# ---------------------------
println("Fitting Gaussian approx via mean/std of samples ...")
p_Gauss = fit(Normal, samples)

println("Running rejection sampling ...")
Nsamp = 10000
RSsamples, M, accept_, reject_ = RejectionSampling.rejectionSampling(Nsamp, q_obs, p_Gauss)

# Save to CSV
df = DataFrame(sample = RSsamples)
#CSV.write("POLYNOMIAL1D_SAMPLES.csv", df)

#
RSsamples_inf, M_inf, accept_inf, reject_inf = RejectionSampling.rejectionSampling(Nsamp, q_inf, p_Gauss)
df_inf = DataFrame(sample = RSsamples_inf)
#CSV.write("POLYNOMIAL1D_SAMPLES_inf.csv", df_inf)

# ---------------------------
# Rejection sampling against mixture Gaussian approx
# ---------------------------
println("Fitting Mixture Gaussian with D/2=4/2=2 mixtures")
K = 5
p_gmm = MixtureModel(GMM(K, samples; method=:kmeans))

println("Running rejection sampling ...")
RSsamples_gmm, M_gmm, accept_gmm, reject_gmm = RejectionSampling.rejectionSampling(Nsamp, q_inf, p_gmm)

# Save to CSV
df_gmm = DataFrame(sample = RSsamples_gmm)
#CSV.write("POLYNOMIAL1D_SAMPLES_gmm.csv", df_gmm)

# ---------------------------
# Plotting
# ---------------------------
 println("plotting . . .")
 common_kwargs = (
     xtickfont = font(4),
     ytickfont = font(4),
     legendfont=font(4),
     legend = :topleft,
     titlefontsize=6,
     alpha=0.5
 )


plt1 = plot(midpts, pdf_vals, title="Discretized observation pdf p_obs=exp(-f_obs)/Z \n θ_obs=$(string(poly_obs.θ))", label="exp(-f_obs)/Z"; common_kwargs...)

plt2 = plot(range, [q_inf(x) for x in range], label="exp(-f_inferred)", title="Unnormalized Inferred Model exp(-f_inferred) \nθ_inferred=$(string(round.(poly_inf.θ; digits=4)))"; common_kwargs...)

plt3 = plot(range, [q_inf(x) for x in range], label="exp(-f_inferred)", title="Inferred Model vs Mixture Gaussian effective model"; common_kwargs...)
plot!(range, [M_gmm*pdf(p_gmm, x) for x in range], label="M·q_gmm(x) \n(M=$M_gmm)")

plt4 = histogram(samples[1:Nsamp], bins=100, label="exp(-f_obs)/Z", title="Histogram of samples\nSample size= $Nsamp each"; common_kwargs...)
histogram!([x[1] for x in RSsamples_gmm] , bins=100, label="exp(-f_inferred)/Z_inferred", alpha=0.5)

plt = plot(plt1, plt2, plt3, plt4, layout=(2,2), dpi=1600)
println("finishing")
savefig(plt, "toy1d.png")
display(plt)

