using GLMakie
import DataFrames, CSV, PairPlots
include("Utils.jl")
using .Utils
using Distributions, Plots, Base.Threads
using Random, CSV, DataFrames
using Graphs
using GraphPlot
using LinearAlgebra
#using PairPlots
#using CairoMakie
using GaussianMixtures
using Images, ImageMagick, ImageTransformations
using CSV, DataFrames

#data_og = Utils.load_data()
#data = Utils.coarse_grain(data_og)
# Example: pick channel 1
#field = data[:, :, :, 9]

df_obs = CSV.read("observation.csv", DataFrame)
df_inf = CSV.read("inferred.csv", DataFrame)
df_gmm = CSV.read("gmm.csv", DataFrame)

# spatial size after coarse-graining
Nx = Ny = Nz = 32   # because 32^3 = 32768
cr = (-1.5, 1.5)

field1 = reshape(df_obs.x1, Nx, Ny, Nz)

field2 = reshape(df_inf.x1, Nx, Ny, Nz)

field3 = reshape(df_gmm.x1, Nx, Ny, Nz)

fig = Figure(resolution = (600, 600))
ax = Axis3(fig[1, 1], title = "Observation", titlesize = 28)

plt = volume!(
 ax,
 field1,
 colorrange = cr
)

save("1_obs.png", fig)


fig = Figure(resolution = (600, 600))
ax = Axis3(fig[1, 1], title = "Model", titlesize = 28)

plt = volume!(
 ax,
 field2,
 colorrange = cr
)

save("1_inf.png", fig)


fig = Figure(resolution = (600, 600))
ax = Axis3(fig[1, 1], title = "GMM", titlesize = 28)

plt = volume!(
 ax,
 field3,
 colorrange = cr
)

save("1_gmm.png", fig)



png_files = [
 "1_obs.png",
 "1_inf.png",
 "1_gmm.png"
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
save("channel1.png", final_img)
