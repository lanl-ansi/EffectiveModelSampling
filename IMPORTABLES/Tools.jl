module Tools

using Distributions, Plots, Base.Threads
using Random, CSV, DataFrames
using Graphs
using GraphPlot
using LinearAlgebra
using PairPlots
using CairoMakie
using GaussianMixtures
using Images, ImageMagick, ImageTransformations

export safe_dataframe, safe_pairplot, adamGD

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
        batch_t = Matrix(obs[rand(1:1000, B), :])
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

end
