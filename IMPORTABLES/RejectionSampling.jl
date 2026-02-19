module RejectionSampling

using Random
using Base.Threads
using Distributions
using GaussianMixtures

export rejectionSampling

# =======================================================
# Estimate M via thread-parallel ratios
# =======================================================
function estimate_ratios_parallel(p_func, q, x_samps)
    ratios_per_thread = [Float64[] for _ in 1:nthreads()]

    @threads for i in 1:length(x_samps)
        tid = threadid()
        x = x_samps[i]
        qx = pdf(q, x)
        if qx > 0.0
            rat = p_func(x) / qx
            if isfinite(rat)
                push!(ratios_per_thread[tid], rat)
            end
        end
    end

    return vcat(ratios_per_thread...)
end

# =======================================================
# Rejection sampling (handles 1D and multivariate)
# =======================================================
function rejectionSampling(N::Int, p, q; num_trials=10^6, checkM=true)
    # Helper: safely compute p(x)
    p_func(x) = try
        pdf(p, x)
    catch
        p(x)
    end

    # Draw samples for estimating M
    println("Estimating M via parallel sampling...")
    x_samps = [rand(q) for _ in 1:num_trials]
    ratios = estimate_ratios_parallel(p_func, q, x_samps)
    M = isempty(ratios) ? 1.0 : maximum(ratios)
    if checkM
        if M > 1000000
            println("Warning: M too large ($M > 100000). Returning empty sample set.")
            top_vals = sort(ratios; rev=true)[1:min(10, length(ratios))]
            println("Top 10 ratio values: ", top_vals)
            D = isa(x_samps[1], Number) ? 1 : length(x_samps[1])
            return Vector{Vector{Float64}}(), M, 0, 0
        end
    end

    println("Estimated M = $M")

    # Thread-local storage for accepted samples
    samples_per_thread = [Vector{Vector{Float64}}() for _ in 1:nthreads()]
    rngs = [MersenneTwister(rand(UInt)) for _ in 1:nthreads()]
    # Thread-local counters
    accepted_per_thread = zeros(Int, nthreads())
    rejected_per_thread = zeros(Int, nthreads())

    # Determine dimension
    D = isa(x_samps[1], Number) ? 1 : length(x_samps[1])

    # Parallel rejection sampling
    @threads for t in 1:nthreads()
        rng = rngs[t]
        local_samples = Vector{Vector{Float64}}()
        n_local = ceil(Int, N / nthreads())

        while length(local_samples) < n_local
            x = rand(rng, q)
            x_vec = isa(x, Number) ? [x] : x  # unify 1D and multivariate
            u = rand(rng)
            px = p_func(x)
            qx = pdf(q, x)
            if qx == 0.0
                continue
            end
            if u < (px / (M * qx))
                push!(local_samples, x_vec)
                accepted_per_thread[t] += 1
                println("samples=", length(local_samples))
            else
                rejected_per_thread[t] += 1
            end
        end
        
        samples_per_thread[t] = local_samples

        if length(local_samples) % 1000 == 0
            @info "Thread $t: collected $(length(local_samples)) / $n_local samples"
        end
    end

    # Combine thread-local samples and trim
    samples = vcat(samples_per_thread...)
    samples = samples[1:N]
    total_accepted = sum(accepted_per_thread)
    total_rejected = sum(rejected_per_thread)
    println("Rejection sampling done: Generated $N samples (dim=$D)")
    println("Total accepted: $total_accepted, Total rejected: $total_rejected")

    return samples, M, total_accepted, total_rejected
end

end # module

