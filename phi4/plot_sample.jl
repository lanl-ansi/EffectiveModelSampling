using CSV, DataFrames, CairoMakie

# Load correct CSV
df = CSV.read("data0.csv", DataFrame)

# Choose 3 samples (e.g., 1, 2, 3)
sample_ids = [1, 1000, 10000]

# Convert to matrices
grids = [reshape(Vector(df[:, id]), 4, 4) for id in sample_ids]

# Compute global min/max across all three samples
global_min = minimum([minimum(g) for g in grids])
global_max = maximum([maximum(g) for g in grids])

# Create figure
fig = Figure(resolution = (1200, 450))
# Store the first heatmap object to attach a shared colorbar
for (i, sid) in enumerate(sample_ids)
    grid = reshape(Vector(df[:, sid]), 4, 4)

    ax = Axis(fig[1, 2*i - 1])
    hm = heatmap!(ax, grid)
    ax.title = "Sample $sid"

    Colorbar(fig[1, 2*i], hm)
end


# Save figure
save("phi4_samples_1_1000_10000.png", fig)

