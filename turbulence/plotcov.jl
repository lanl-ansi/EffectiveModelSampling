using CSV, DataFrames, Statistics, Plots

# Read data
df_obs = CSV.read("observation.csv", DataFrame)
df_inf = CSV.read("inferred.csv", DataFrame)
df_gmm = CSV.read("gmm.csv", DataFrame)

# Compute covariance matrices
cov_obs = cov(Matrix(df_obs))
cov_inf = cov(Matrix(df_inf))
cov_gmm = cov(Matrix(df_gmm))

# Determine global color limits for consistent scaling
all_cov = [cov_obs; cov_inf; cov_gmm]
vmin, vmax = minimum(all_cov), maximum(all_cov)
# Function to make a covariance heatmap
function cov_heatmap(cov_matrix, title_str; vmin=vmin, vmax=vmax)
    return heatmap(cov_matrix,
        title=title_str,
        color=:PiYG,       # diverging color for clear positive/negative
        clims=(vmin, vmax), # same color scale across plots
        size=(300,300)
        #annotate=cov_matrix # show numbers on cells
    )
end

# Create individual heatmaps
p1 = cov_heatmap(cov_obs, "Observation")
p2 = cov_heatmap(cov_inf, "Model")
p3 = cov_heatmap(cov_gmm, "GMM")

# Combine them side by side
combined = plot(p1, p2, p3, layout=(1,3), size=(1200,400))

# Save to file
savefig(combined, "cov.png")
