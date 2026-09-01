module ImportanceSampling

using Random
using Base.Threads
using Distributions
using GaussianMixtures

export importanceSampling

function importanceSampling(N::Int, p, q, h; verbose=true)
    N > 0 || throw(ArgumentError("N must be positive"))

    samples = Vector{Any}(undef, N)
    weights = Vector{Float64}(undef, N)
    contributions = Vector{Float64}(undef, N)

    # One independent RNG per thread
    rngs = [
        MersenneTwister(rand(UInt))
        for _ in 1:nthreads()
    ]

    @threads for i in 1:N
        rng = rngs[threadid()]

        # Draw X_i ~ q
        x = rand(rng, q)

        # Evaluate unnormalized p and normalized proposal q
        px = p(x)
        qx = pdf(q, x)

        qx > 0.0 || throw(
            DomainError(x, "q(x) must be positive at every sampled point")
        )

        weight = px / qx
        contribution = weight * h(x)

        isfinite(weight) || throw(
            DomainError(weight, "Non-finite importance weight")
        )

        isfinite(contribution) || throw(
            DomainError(contribution, "Non-finite importance contribution")
        )

        samples[i] = x
        weights[i] = weight
        contributions[i] = contribution
    end

    # I_N[h] = (1/N) sum_i [p(X_i)/q(X_i)] h(X_i)
    I_N = mean(contributions)

    if verbose
        D = samples[1] isa Number ? 1 : length(samples[1])

        println("Importance sampling done")
        println("Number of proposal samples: $N")
        println("Dimension: $D")
        println("I_N[h] = $I_N")
    end

    return I_N #(
        #estimate = I_N,
        #samples = samples,
        #weights = weights,
        #contributions = contributions,
    #)
end


end # module

