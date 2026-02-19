# score matching uing ADAM
module ADAM_ScoreMatching

using Random, Printf, Base.Threads, Statistics, LinearAlgebra
include("MultiDNomial.jl")
using .MultiDNomial
export adam_score_matching!, grad_score_matching, unnormalize_params!

#########################################
# unnormalizing params
#########################################
# Put this in your MultiDNomial.jl (or a helper file you include)
# Usage: unnormalize_params!(f, mean_x_vec, std_x_vec)
function unnormalize_params!(f::PolynomialModel, mean_x::AbstractVector{<:Real}, std_x::AbstractVector{<:Real})
    D = f.D
    if length(mean_x) != D || length(std_x) != D
        throw(ArgumentError("mean_x and std_x must have length equal to f.D"))
    end

    # convert to Float64 vectors
    μ = Float64.(mean_x)
    σ = Float64.(std_x)

    # global accumulator: maps exponent-tuple (like (2,0,1)) => coefficient
    acc = Dict{Tuple{Vararg{Int}}, Float64}()

    # iterate over current (normalized) basis terms
    for (i, α_vec) in enumerate(f.α)
        θi = Float64(f.θ[i])
        # start with single term: zero exponent tuple -> coeff θi
        exps = Dict{Tuple{Vararg{Int}}, Float64}()
        exps[Tuple(ntuple(_ -> 0, D))] = θi

        # for each dimension expand ( (x_d - μ_d)/σ_d )^αd
        for d in 1:D
            αd = α_vec[d]
            if αd == 0
                # nothing to do for this dim, exps unchanged
                continue
            end
            new_exps = Dict{Tuple{Vararg{Int}}, Float64}()

            # for each partial expansion so far, multiply by expansion along dim d
            for (exp_t, coeff_so_far) in exps
                # exp_t is a tuple of length D
                for k in 0:αd
                    # coefficient contribution from this dimension:
                    # binom(αd,k) * (-μ_d)^(αd-k) / σ_d^αd
                    cd = binomial(αd, k) * ((-μ[d])^(αd - k)) / (σ[d]^αd)
                    # build new exponent tuple with k added at dimension d
                    exp_list = collect(exp_t)
                    exp_list[d] += k
                    exp_t_new = Tuple(exp_list)
                    new_exps[exp_t_new] = get(new_exps, exp_t_new, 0.0) + coeff_so_far * cd
                end
            end

            exps = new_exps
        end

        # now exps contains all contributions (x^β) from the original θi * normalized monomial
        for (β_t, c) in exps
            acc[β_t] = get(acc, β_t, 0.0) + c
        end
    end

    # Build new α and θ lists from accumulator (sorted for determinism)
    keys_list = collect(keys(acc))
    # sort by total degree then lexicographic so result is deterministic (optional)
    sort!(keys_list, by = t -> (sum(t), t))

    α_new = [collect(t) for t in keys_list]
    θ_new = [acc[t] for t in keys_list]

    # Replace f's basis & coefficients in-place
    f.α = α_new
    f.θ = θ_new

    return f
end


#########################################################
# Gradient for score matching
#########################################################
function grad_score_matching(f::PolynomialModel, x_samps::Vector{Vector{Float64}}, z_precom)
    N = length(x_samps)
    D = f.D
    K = length(f.α)
    nthreads = Threads.nthreads()
    grads_thread = [zeros(Float64, K) for _ in 1:nthreads]

    Threads.@threads for i in 1:N
        tid = Threads.threadid()
        local_grad = grads_thread[tid]

        x_i = x_samps[i]
        powers, df_dxj_matrix, d2f_dx2j_matrix = z_precom[i]#precompute_monomials(f, x_i)

        for j in 1:D
            for k in 1:K
                local_grad[k] += df_dxj(f, df_dxj_matrix, j) * df_dxjdθα(f, df_dxj_matrix, k, j) -
                                 hess_df_dθα(f, d2f_dx2j_matrix, k, j)
            end
        end
    end

    # Combine threads
    grad = reduce(+, grads_thread) ./ N
    return grad
end
#########################################################
# ADAM step
#########################################################
mutable struct AdamParam
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

function adam_step!(θ, grad, state::AdamParam; η=0.01, β1=0.9, β2=0.999, ϵ=1e-8, clip_grad=Inf)
    state.t += 1

    # Gradient clipping if requested
    if clip_grad < Inf
        norm_grad = norm(grad)
        if norm_grad > clip_grad
            grad = (clip_grad / norm_grad) .* grad
        end
    end

    state.m .= β1 .* state.m .+ (1 - β1) .* grad
    state.v .= β2 .* state.v .+ (1 - β2) .* (grad .^ 2)

    mhat = state.m ./ (1 .- β1^state.t)
    vhat = state.v ./ (1 .- β2^state.t)

    θ .-= η .* mhat ./ (sqrt.(vhat) .+ ϵ)
end

#########################################################
# Adam-based score matching training
#########################################################
function adam_score_matching!(f::PolynomialModel, xs::AbstractVector{<:AbstractVector{<:Real}};
                              η::Float64=0.01,
                              β1::Float64=0.9,
                              β2::Float64=0.999,
                              ϵ::Float64=1e-8,
                              tol_loss::Float64=1e-6,
                              batch_size::Int=1000,
                              max_epochs::Int=5000,
                              verbose::Bool=true,
                              clip_grad::Float64=Inf)


    N = length(xs)
    D = length(xs[1])
    K = length(f.α)

    # ------------------------
    # Normalize samples
    # ------------------------
    mat = reduce(hcat, xs)'          # N x D matrix
    mean_x = mapslices(mean, mat; dims=1)[:]   # vector length D
    std_x  = mapslices(std,  mat; dims=1)[:]
    std_x[std_x .== 0.0] .= 1.0                  # avoid divide by zero

    z_samps = [(x .- mean_x) ./ std_x for x in xs]  # normalized samples
    println("First few z_samps: ", first(z_samps, 10))
    @time z_precomp = [precompute_monomials(f, z) for z in z_samps]

    # ------------------------
    # Initialize Adam state
    # ------------------------
    state = AdamParam(zeros(K), zeros(K), 0)
    
    # Cosine decay learning rate schedule
    η_schedule(epoch; η0=η) = η0 * 0.5 * (1 + cos(π * epoch / max_epochs))

    epoch = 0
    loss_prev = Inf

    while epoch < max_epochs
        epoch += 1

        # Shuffle samples each epoch
        shuffled_idxs = randperm(N)
        z_samps_shuffled = z_samps[shuffled_idxs]

        for batch_start in 1:batch_size:N
            batch_end = min(batch_start + batch_size - 1, N)
            batch = z_samps_shuffled[batch_start:batch_end]

            grad_batch = grad_score_matching(f, batch, z_precomp[shuffled_idxs[batch_start:batch_end]])
            η_t = η_schedule(epoch)
            adam_step!(f.θ, grad_batch, state; η=η_t, β1=β1, β2=β2, ϵ=ϵ, clip_grad=clip_grad)
            #println("epoch=", epoch, "curr f.θ=", f.θ)
        end

        # ------------------------
        # Optional loss monitoring
        # ------------------------
        if verbose && (epoch % 50 == 0 || epoch == 1)
            loss_val = sum([0.5 * sum(df_dxj(f, precompute_monomials(f, z)[2], j)^2 for j in 1:f.D) -
                sum(hess_df_dθα(f, precompute_monomials(f, z)[3], k, j) for j in 1:f.D, k in 1:K)
                for z in z_samps]) / N
            #@printf("Epoch %d, Score Matching Loss = %.6f, theta = %s\n", epoch, loss_val, string(f.θ))
            @printf("Epoch %d, Score Matching Loss = %.6f\n", epoch, loss_val)

            # Convergence check
            if abs((loss_val - loss_prev)/loss_prev) < tol_loss
                println("Converged at epoch $epoch, loss change < $tol_loss")
                break
            end
            loss_prev = loss_val
        end
    end


    # ------------------------
    # Un-normalize parameters to original x-scale
    # ------------------------
    unnormalize_params!(f, mean_x, std_x)
    println("final f.\theta=", f.θ)

    return f
end



end









