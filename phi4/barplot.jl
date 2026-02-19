using Statistics
using DataFrames
using Plots

# --------- Read file ---------
content = read("toplot_mosum.txt", String)

# --------- Split into blocks ---------
blocks = split(content, r"FILE HEADER:\s*")
blocks = filter(b -> !isempty(strip(b)), blocks)

# --------- Initialize DataFrame ---------
df_gmm = DataFrame(
    dataset = Int[],
#    method  = String[],
    err1    = Float64[],
    err2    = Float64[],
    err3    = Float64[],
    err4    = Float64[],
)

df_inf = DataFrame(
    dataset = Int[],
#    method  = String[],
    err1    = Float64[],
    err2    = Float64[],
    err3    = Float64[],
    err4    = Float64[],
)
# --------- Parse each block ---------
for block in blocks
    # Extract dataset index
    m = match(r"4moments_(\d+)\.txt", block)
    m === nothing && continue
    dataset = parse(Int, m.captures[1])

    # Extract errors
    gmm_vals = parse.(Float64,
        eachmatch(r"err_gmm\d+\s*=\s*([0-9.eE+-]+)", block)
        .|> x -> x.captures[1]
    )
    inf_vals = parse.(Float64,
        eachmatch(r"err_inf\d+\s*=\s*([0-9.eE+-]+)", block)
        .|> x -> x.captures[1]
    )

    length(gmm_vals) == 4 || error("Dataset $dataset does not have 4 GMM errors")
    length(inf_vals) == 4 || error("Dataset $dataset does not have 4 INF errors")

    # --------- Push rows ---------
    push!(df_gmm, (dataset, gmm_vals...))
    push!(df_inf, (dataset, inf_vals...))
end
"""
#calculate mean + std

#means_gmm = [mean(df_gmm.err1), mean(df_gmm.err2), mean(df_gmm.err3), mean(df_gmm.err4)]
#means_inf = [mean(df_inf.err1), mean(df_inf.err2), mean(df_inf.err3), mean(df_inf.err4)]
#stds_gmm  = [std(df_gmm.err1),  std(df_gmm.err2),  std(df_gmm.err3),  std(df_gmm.err4)]
#stds_inf  = [std(df_inf.err1),  std(df_inf.err2),  std(df_inf.err3),  std(df_inf.err4)]

plt1 = bar(["err1", "err2", "err3", "err4"],
    means_gmm,
    yerror = stds_gmm,
    label = "GMM",
    xlabel = "Means and standard deviation",
    ylabel = "Error of moment",
    alpha = 0.8,
    bar_width = 0.5
#    legend = false,
#    legend = :topright
)

plt2 = bar(["err1", "err2", "err3", "err4"],
    means_inf,
    yerror = stds_inf,
    label = "INF", 
    #xlabel = "Means and std",
    #ylabel = "Moment",
    alpha = 0.8,
    #label = "INF",
    xlabel = "Means and standard deviation",
    ylabel = "Error of moment",
    bar_width = 0.5
#    legend = false
)

"""
# ---- compute means + stds ----
gmm_means = [mean(df_gmm.err1), mean(df_gmm.err2), mean(df_gmm.err3), mean(df_gmm.err4)]
gmm_stds  = [std(df_gmm.err1),  std(df_gmm.err2),  std(df_gmm.err3),  std(df_gmm.err4)]

inf_means = [mean(df_inf.err1), mean(df_inf.err2), mean(df_inf.err3), mean(df_inf.err4)]
inf_stds  = [std(df_inf.err1),  std(df_inf.err2),  std(df_inf.err3),  std(df_inf.err4)]

x = 1:4
offset = 0.2

p = bar(
    x .- offset,             # INF bars shifted left
    inf_means,
    yerror = inf_stds,
    bar_width = 0.33,
    alpha = 0.7,
    label = "Model"
)

bar!(
    x .+ offset,             # GMM bars shifted right
    gmm_means,
    yerror = gmm_stds,
    bar_width = 0.33,
    alpha = 0.7,
    label = "GMM"
)

xticks!(x, ["1", "2", "3", "4"])
xlabel!("Moment")
ylabel!("RMSE")
#title!("INF vs GMM  means and standard deviation (side-by-side per error index)")

savefig(p, "side_by_side_barplot.png")
println("Saved to side_by_side_barplot.png")
