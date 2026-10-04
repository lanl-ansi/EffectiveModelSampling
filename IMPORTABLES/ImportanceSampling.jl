module ImportanceSampling

using Random
using Base.Threads
using Distributions
using GaussianMixtures

export importanceSampling, importanceSamplingMoments, importanceSamplingTensorMoments

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



"""
    importanceSamplingTensorMoments(N, p, q; self_normalize=true, verbose=true)
 
Full-tensor version of `importanceSamplingMoments` for a vector-valued X in R^D.
Estimates the 1st-4th raw moment TENSORS of the target `p` from ONE set of N draws
X_i ~ q (the same samples and weights are reused for every order):
 
    m1[a]       = E[X_a]
    m2[a,b]     = E[X_a X_b]
    m3[a,b,c]   = E[X_a X_b X_c]
    m4[a,b,c,d] = E[X_a X_b X_c X_d]
 
Shapes and index convention are the same as `Moments.firstMoment ... fourthMoment`, so the
results can be compared directly with `Moments.getErr`. (`importanceSamplingMoments` only
returns the elementwise powers E[X_a^k], i.e. the diagonal entries m_k[a,...,a].)
 
Arguments (target FIRST, proposal SECOND):
- `p(x)`: target density, may be unnormalized; takes a Vector and returns a number >= 0
- `q`   : proposal, a multivariate Distributions.jl distribution (e.g. a `MixtureModel`)
 
- `self_normalize=true`:  m_k = sum_i w_i X_i^(x)k / sum_i w_i   (use when `p` is only known up to a constant)
- `self_normalize=false`: m_k = (1/N) * sum_i w_i X_i^(x)k       (use only when `p` is normalized)
 
where w_i = p(X_i) / q(X_i). Weights are computed in log-space for numerical stability.
 
Returns a NamedTuple `(m1, m2, m3, m4, ess)`. `ess = (sum w)^2 / sum(w^2)` is the effective
sample size of the weights; report it next to the moments. A warning is issued when it drops
below 1% of N, because the estimate is then dominated by a handful of samples.
 
Memory: two N x D^2 matrices plus the D^4 tensor m4.
"""


function importanceSamplingTensorMoments(N::Int, p, q; self_normalize::Bool=true, verbose::Bool=true)
    N > 0 || throw(ArgumentError("N must be positive"))
 
    # One draw just to learn the dimension
    x1 = rand(q)
    x1 isa AbstractVector || throw(ArgumentError(
        "rand(q) must return a vector; for scalar X use importanceSamplingMoments"
    ))
    D = length(x1)
 
    X    = Matrix{Float64}(undef, N, D)   # row i is X_i
    logw = Vector{Float64}(undef, N)
 
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
        lq = logpdf(q, x)
 
        isfinite(lq) || throw(
            DomainError(x, "q(x) must be positive at every sampled point")
        )
        (px >= 0 && isfinite(px)) || throw(
            DomainError(px, "p(x) must be finite and non-negative")
        )
 
        X[i, :] = x
        logw[i] = log(px) - lq            # log w_i = log p(X_i) - log q(X_i)
    end
 
    logw_max = maximum(logw)
    isfinite(logw_max) || throw(
        DomainError(logw_max, "p(x) is zero at every sampled point")
    )
 
    if self_normalize
        # Shift by max(log w) before exponentiating; the shift cancels in the normalization.
        w = exp.(logw .- logw_max)
        w ./= sum(w)                      # sum_i w_i = 1
    else
        w = exp.(logw) ./ N               # plain estimator: p must be normalized
        all(isfinite, w) || throw(
            DomainError(maximum(w), "Non-finite importance weight")
        )
    end
 
    ess      = sum(w)^2 / sum(abs2, w)
    w_maxrel = maximum(w) / sum(w)
 
    # Pairwise products X_a * X_b, stored in column a + D*(b-1). Reshaping the products
    # below then gives m2[a,b], m3[a,b,c], m4[a,b,c,d] directly (column-major order).
    X2 = Matrix{Float64}(undef, N, D * D)
    for b in 1:D, a in 1:D
        @views X2[:, a + D * (b - 1)] .= X[:, a] .* X[:, b]
    end
 
    Xw = X .* w                           # row i scaled by w_i
 
    m1 = vec(sum(Xw; dims=1))
    m2 = reshape(X2' * w,         D, D)
    m3 = reshape(X2' * Xw,        D, D, D)
    m4 = reshape(X2' * (X2 .* w), D, D, D, D)
 
    if verbose
        println("Importance sampling (tensor moments) done")
        println("Number of proposal samples: $N")
        println("Dimension: $D")
        println("Self-normalized: $self_normalize")
        println("Effective sample size: $(round(ess; digits=1)) ($(round(100 * ess / N; digits=2))% of N)")
        println("Largest normalized weight: $(round(w_maxrel; sigdigits=3))")
        self_normalize || println("sum_i w_i = $(sum(w))  (should be close to 1 if p is normalized)")
    end
 
    ess / N < 0.01 && @warn "Effective sample size is below 1% of N; the IS moments are unreliable" ess N
 
    return (m1=m1, m2=m2, m3=m3, m4=m4, ess=ess, n=N, self_normalize=self_normalize)
end


end # module

