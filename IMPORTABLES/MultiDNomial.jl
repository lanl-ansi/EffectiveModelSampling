# dealing with creating f and its derivatives
module MultiDNomial
using LinearAlgebra
export PolynomialModel, monomial, df_dxj, d2f_dx2j, df_dxjdθα, df_dθα, hess_df_dθα, precompute_monomials


#########################################
# helper bois
#########################################
function make_α(D::Int, L::Int)
    """
    Generate all multi-indices a = (a1,...,aD) with total degree <= L.
    """
    indices = Tuple{Vararg{Int,D}}[]
    function rec_build(prefix, left, pos)
        if pos > D
            push!(indices, tuple(prefix...))
        else
            for v in 0:left
                rec_build([prefix... , v], left - v, pos+1)
            end
        end
    end
    rec_build(Int[], L, 1)
    # convert tuples to vectors
    αs= [collect(t) for t in indices]
    return αs#[α for α in αs if any(α .> 0)] #removing the constant term
end

mutable struct PolynomialModel
    D::Int #dimension
    L::Int #degree
    α::Vector{Vector{Int}}
    θ::Vector{Float64} 
end

function PolynomialModel(D::Int, L::Int)
    # D: dimension
    # L: degree
    α = make_α(D,L) 
    θ = [randn() * (0.1 / (1 + sum(a))) for a in α]
    return PolynomialModel(D, L, α, θ)
end

function monomial(x,α)
    """
    x^α
    """
    prod(x[d]^α[d] for d in 1:length(x)) 
end

function powers_for_sample(f::PolynomialModel, x::AbstractVector)
    """
    Precompute x.^α_k for all monomials α_k in f.α
    Returns a vector of length K = length(f.α)
    """
    return [monomial(x, α) for α in f.α]
end

function monomial_fast(x::Vector{Float64}, α::Vector{Int}, x_powers::Vector{Vector{Float64}})
    # x_powers[d][k] = (x[d])^k
    prod(x_powers[d][α[d]+1] for d in 1:length(x))  # +1 since powers start at ^0
end

function precompute_powers(x::Vector{Float64}, L::Int)
    D = length(x)
    [ [x[d]^k for k in 0:L] for d in 1:D ]
end

function precompute_monomials(f::PolynomialModel, x::AbstractVector)
    """
    Precompute all monomials and derivatives needed for score matching:
    - powers[k] = x^α_k
    - df_dxj_matrix[j,k] = ∂(x^α_k)/∂x_j
    - d2f_dx2j_matrix[j,k] = ∂²(x^α_k)/∂x_j²
    """
    K = length(f.α)
    D = f.D
    x_powers = precompute_powers(x, f.L)

    powers = zeros(K)
    df_dxj_matrix = zeros(D, K)
    d2f_dx2j_matrix = zeros(D, K)

    for k in 1:K
        α = f.α[k]
        powers[k] = monomial_fast(x, α, x_powers)

        for j in 1:D
            if α[j] > 0
                α_minus_e = copy(α); α_minus_e[j] -= 1
                df_dxj_matrix[j,k] = α[j] * monomial_fast(x, α_minus_e, x_powers)
            end
            if α[j] >= 2
                α_minus_2e = copy(α); α_minus_2e[j] -= 2
                d2f_dx2j_matrix[j,k] = α[j]*(α[j]-1)*monomial_fast(x, α_minus_2e, x_powers)
            end
        end
    end
    return powers, df_dxj_matrix, d2f_dx2j_matrix
end



#########################################
# function f and its gradients 
#########################################
function (f::PolynomialModel)(x)
    """
    Function f of dimension D and degree L
    """
    return sum(f.θ[i]*monomial(x,f.α[i]) for i in 1:length(f.α))
end

function df_dxj(f::PolynomialModel, df_dxj_matrix, j)
    """
    ∂f/∂dx_j 
    """
    return sum(f.θ .* df_dxj_matrix[j, :])
end

function d2f_dx2j(f::PolynomialModel, x, j)
    """
    ∂^2f/∂dx^2_j 
    """
    s = 0.0
    for (i, α) in enumerate(f.α) 
        αj = α[j]
        if αj≥2
            e_j = zeros(Int, length(α))
            e_j[j] = 2
            term = αj*(αj-1)*monomial(x, α.-e_j)
            s+=f.θ[i]*term
        end
    end
    return s
end

function df_dxjdθα(f::PolynomialModel, df_dxj_matrix, k, j)
    """
    Uses precomputed powers vector.
    powers[k] == monomial(x, α_k)
    """
    return df_dxj_matrix[j, k]
end

function df_dθα(f::PolynomialModel, x, k)
    """
    ∂f/∂dθ_{α_k} 
    """
    return monomial(x,f.α[k]) 
end

function hess_df_dθα(f::PolynomialModel, d2f_dx2j_matrix, k, j)
    """
    ∂^2/∂x^2_j (∂f/∂dθ_α)
    """
    return d2f_dx2j_matrix[j, k]
end


end
