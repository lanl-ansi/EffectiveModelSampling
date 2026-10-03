module ImportanceSampling

using Random
using Base.Threads
using Distributions
using GaussianMixtures

export importanceSampling, importanceSamplingMoments

function importanceSampling(N::Int, p, q, h; verbose=true)
    N > 0 || throw(ArgumentError("N must be positive"))

    samples = Vector{Any}(undef, N)
    weights = Vector{Float64}(undef, N)
    contributions = Vector{Float64}(undef, N)

    # One independent RNG per thread
    rngs = [
        MersenneTwister(rand(UInt))
        for _ in 1:Threads.maxthreadid()
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


"""
    importanceSamplingMoments(N, p, q; self_normalize=false, verbose=true)
 
Estimate the 2nd, 3rd and 4th raw moments E[X^2], E[X^3], E[X^4] of the target `p`
from ONE set of N draws X_i ~ q (the same samples and weights are reused for all
three moments). For vector-valued X the powers are taken elementwise.
 
- `self_normalize=false`: m_k = (1/N) * sum_i w_i * X_i^k            (use when `p` is normalized)
- `self_normalize=true`:  m_k = sum_i w_i * X_i^k / sum_i w_i        (use when `p` is only known up to a constant)
 
where w_i = p(X_i) / q(X_i).
 
Returns a NamedTuple `(m2, m3, m4)`.
"""
function importanceSamplingMoments(N::Int, p, q; self_normalize::Bool=false, verbose::Bool=true)
    N > 0 || throw(ArgumentError("N must be positive"))
 
    samples = Vector{Any}(undef, N)
    weights = Vector{Float64}(undef, N)
 
    # One independent RNG per thread
    rngs = [
        MersenneTwister(rand(UInt))
        for _ in 1:Threads.maxthreadid()
    ]
 
    @threads for i in 1:N
        rng = rngs[threadid()]
 
        # Draw X_i ~ q
        x = rand(rng, q)
 
        px = p(x)
        qx = pdf(q, x)
 
        qx > 0.0 || throw(
            DomainError(x, "q(x) must be positive at every sampled point")
        )
 
        weight = px / qx
 
        isfinite(weight) || throw(
            DomainError(weight, "Non-finite importance weight")
        )
 
        samples[i] = x
        weights[i] = weight
    end
 
    denom = self_normalize ? sum(weights) : N
 
    # sum_i w_i * X_i^k / denom  (elementwise power if X is a vector)
    rawmoment(k) = sum(weights[i] .* samples[i] .^ k for i in 1:N) ./ denom
    m1 = rawmoment(1) 
    m2 = rawmoment(2)
    m3 = rawmoment(3)
    m4 = rawmoment(4)
 
    if verbose
        D = samples[1] isa Number ? 1 : length(samples[1])
 
        println("Importance sampling (moments) done")
        println("Number of proposal samples: $N")
        println("Dimension: $D")
        println("Self-normalized: $self_normalize")
        println("E[X] ≈ $m1")
        println("E[X^2] ≈ $m2")
        println("E[X^3] ≈ $m3")
        println("E[X^4] ≈ $m4")
    end
 
    return (m1=m1, m2 = m2, m3 = m3, m4 = m4)
end


end # module

