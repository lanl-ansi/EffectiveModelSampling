module EffectiveModel

using Random
using Base.Threads
using Distributions
using GaussianMixtures
using LinearAlgebra

export effectiveModel
println("THREAD NUM REJECTION SAMPLING=", Threads.nthreads())

function effectiveModel(K::Int, samples)
    if length(size(samples))==1
        p_gmm = MixtureModel(GMM(K, samples; method=:kmeans))
    else
        p_gmm = MixtureModel(GMM(K, samples; method=:kmeans, kind=:full))

        #check that covariance parameters are finite
        for k in 1:K
            Σ = p_gmm.components[k].Σ

            # Check that all covariance entries are finite
            @assert all(isfinite, Σ) "Covariance matrix $k contains NaN or Inf."

            # Check symmetry
            @assert isapprox(Σ, Σ'; atol=1e-10) "Covariance matrix $k is not symmetric."

            # Check positive definiteness
            @assert isposdef(Σ) "Covariance matrix $k is not positive definite."
        end
    end

    return p_gmm
end

end # module

