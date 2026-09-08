module MultiDNomial

using LinearAlgebra
using Random

export PolynomialModel,
       monomial,
       df_dxj,
       d2f_dx2j,
       df_dxjdθα,
       df_dθα,
       hess_df_dθα,
       precompute_monomials,
       enforce_energy_constraints!,
       energy_constraints_satisfied,
       fixed_coefficient_mask,
       zero_constant!


const DEFAULT_COERCIVITY_FLOOR = 1.0e-6


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

"""
    make_α(D, L)

Generate all multi-indices α in N_0^D with total degree at most L.
"""
function make_α(D::Int, L::Int)
    D > 0 || throw(ArgumentError("D must be positive"))
    L >= 0 || throw(ArgumentError("L must be nonnegative"))

    indices = Tuple{Vararg{Int,D}}[]

    function rec_build!(
        prefix::Vector{Int},
        degree_left::Int,
        position::Int,
    )
        if position > D
            push!(indices, Tuple(prefix))
            return
        end

        for exponent in 0:degree_left
            push!(prefix, exponent)
            rec_build!(prefix, degree_left - exponent, position + 1)
            pop!(prefix)
        end
    end

    rec_build!(Int[], L, 1)
    return [collect(index) for index in indices]
end


mutable struct PolynomialModel
    D::Int
    L::Int
    α::Vector{Vector{Int}}
    θ::Vector{Float64}
end


"""
    PolynomialModel(D, L; rng=Random.default_rng(),
                    coercivity_floor=1e-6)

Create a complete total-degree polynomial model. The maximum degree L must be
positive and even. Every monomial of total degree at most L is retained.

The constructor fixes the constant coefficient at zero and makes the pure
degree-L coefficients large enough to dominate the possibly signed mixed
degree-L coefficients. This guarantees a coercive leading homogeneous
polynomial without deleting interaction terms.
"""
function PolynomialModel(
    D::Int,
    L::Int;
    rng::AbstractRNG=Random.default_rng(),
    coercivity_floor::Real=DEFAULT_COERCIVITY_FLOOR,
)
    D > 0 || throw(ArgumentError("D must be positive"))
    L >= 2 || throw(ArgumentError("L must be at least 2"))
    iseven(L) || throw(
        ArgumentError("L must be even to guarantee an even coercive leading term"),
    )
    _validate_coercivity_floor(coercivity_floor)

    α = make_α(D, L)
    θ = [
        randn(rng) * (0.1 / (1 + sum(exponents)))
        for exponents in α
    ]

    model = PolynomialModel(D, L, α, θ)
    enforce_energy_constraints!(
        model;
        coercivity_floor=coercivity_floor,
    )
    return model
end


# ---------------------------------------------------------------------------
# Coercivity and identifiability constraints
# ---------------------------------------------------------------------------

_is_constant(α::AbstractVector{<:Integer}) = all(iszero, α)

_is_leading(
    α::AbstractVector{<:Integer},
    L::Integer,
) = sum(α) == L

function _is_pure_leading(
    α::AbstractVector{<:Integer},
    L::Integer,
)
    return count(exponent -> exponent != 0, α) == 1 &&
           any(exponent -> exponent == L, α)
end

function _pure_leading_indices(model::PolynomialModel)
    indices = fill(0, model.D)

    for (k, α) in pairs(model.α)
        if _is_leading(α, model.L) &&
           _is_pure_leading(α, model.L)
            coordinate = findfirst(exponent -> exponent == model.L, α)
            coordinate === nothing && error(
                "internal error while locating a pure leading monomial",
            )
            indices[coordinate] == 0 || throw(
                ArgumentError("duplicate pure leading monomial in model.α"),
            )
            indices[coordinate] = k
        end
    end

    all(index -> index != 0, indices) || throw(
        ArgumentError(
            "model.α must contain x_j^L for every coordinate j",
        ),
    )
    return indices
end

function _required_pure_coefficients(
    model::PolynomialModel,
    coercivity_floor::Real,
)
    required = fill(Float64(coercivity_floor), model.D)

    for (coefficient, α) in zip(model.θ, model.α)
        if _is_leading(α, model.L) &&
           !_is_pure_leading(α, model.L)
            magnitude = abs(coefficient)
            for j in 1:model.D
                required[j] +=
                    magnitude * (α[j] / model.L)
            end
        end
    end

    return required
end

function _validate_coercivity_floor(coercivity_floor::Real)
    isfinite(coercivity_floor) || throw(
        ArgumentError("coercivity_floor must be finite"),
    )
    coercivity_floor > 0 || throw(
        ArgumentError("coercivity_floor must be strictly positive"),
    )
    return nothing
end

function _validate_model_structure(model::PolynomialModel)
    model.D > 0 || throw(ArgumentError("model.D must be positive"))
    model.L >= 2 && iseven(model.L) || throw(
        ArgumentError("model.L must be an even integer of at least 2"),
    )
    length(model.α) == length(model.θ) || throw(
        DimensionMismatch("model.α and model.θ must have the same length"),
    )

    expected_terms = binomial(model.D + model.L, model.L)
    length(model.α) == expected_terms || throw(
        DimensionMismatch(
            "the complete degree-$(model.L) basis in dimension " *
            "$(model.D) must contain $expected_terms terms",
        ),
    )

    seen = Set{Tuple{Vararg{Int}}}()
    constant_count = 0

    for α in model.α
        length(α) == model.D || throw(
            DimensionMismatch("every multi-index must have length model.D"),
        )
        all(exponent -> exponent >= 0, α) || throw(
            ArgumentError("multi-index exponents must be nonnegative"),
        )
        sum(α) <= model.L || throw(
            ArgumentError("multi-index degree cannot exceed model.L"),
        )

        key = Tuple(α)
        key in seen && throw(
            ArgumentError("model.α contains a duplicate multi-index"),
        )
        push!(seen, key)
        constant_count += _is_constant(α)
    end

    constant_count == 1 || throw(
        ArgumentError("model.α must contain exactly one constant monomial"),
    )
    all(isfinite, model.θ) || throw(
        DomainError(model.θ, "all polynomial coefficients must be finite"),
    )

    # This also checks that every pure x_j^L term occurs exactly once.
    _pure_leading_indices(model)
    return nothing
end


"""
    fixed_coefficient_mask(model)

Return a mask for coefficients fixed at zero. Only the constant coefficient
is fixed; every nonconstant basis term, including every mixed degree-L term,
remains trainable.
"""
function fixed_coefficient_mask(model::PolynomialModel)
    mask = falses(length(model.α))

    for (k, α) in pairs(model.α)
        mask[k] = _is_constant(α)
    end

    return mask
end


"""
    zero_constant!(model)

Fix the score-unidentifiable constant coefficient at zero.
"""
function zero_constant!(model::PolynomialModel)
    for (k, α) in pairs(model.α)
        if _is_constant(α)
            model.θ[k] = 0.0
        end
    end
    return model
end


"""
    enforce_energy_constraints!(model; coercivity_floor=1e-6)

Keep every nonconstant polynomial term and impose the sufficient coercivity
condition

    theta_(L e_j) >= coercivity_floor
        + sum_{|alpha|=L, alpha mixed}
              abs(theta_alpha) * alpha_j / L.

Weighted AM-GM gives

    abs(x^alpha) <= sum_j (alpha_j / L) * abs(x_j)^L,

so the leading homogeneous polynomial obeys

    f_L(x) >= coercivity_floor * sum_j abs(x_j)^L.

For even L this proves coercivity. Adam must call this function after every
parameter update; enforcing it only at initialization is insufficient.
"""
function enforce_energy_constraints!(
    model::PolynomialModel;
    coercivity_floor::Real=DEFAULT_COERCIVITY_FLOOR,
)
    _validate_coercivity_floor(coercivity_floor)
    _validate_model_structure(model)
    pure_indices = _pure_leading_indices(model)
    required = _required_pure_coefficients(
        model,
        coercivity_floor,
    )
    all(isfinite, required) || throw(
        OverflowError(
            "coercivity bounds overflowed; rescale the coefficients or data",
        ),
    )

    # Mutate only after every validation above has succeeded.
    zero_constant!(model)
    for j in 1:model.D
        index = pure_indices[j]
        model.θ[index] = max(model.θ[index], required[j])
    end

    return model
end


"""
    energy_constraints_satisfied(model;
                                 coercivity_floor=1e-6,
                                 atol=1e-12)

Return true when the constant is zero and the pure degree-L coefficients
dominate all mixed degree-L coefficients by at least coercivity_floor.
"""
function energy_constraints_satisfied(
    model::PolynomialModel;
    coercivity_floor::Real=DEFAULT_COERCIVITY_FLOOR,
    atol::Real=1.0e-12,
)
    _validate_coercivity_floor(coercivity_floor)
    isfinite(atol) && atol >= 0 || throw(
        ArgumentError("atol must be finite and nonnegative"),
    )

    try
        _validate_model_structure(model)
    catch
        return false
    end

    for (coefficient, α) in zip(model.θ, model.α)
        if _is_constant(α) && abs(coefficient) > atol
            return false
        end
    end

    pure_indices = _pure_leading_indices(model)
    required = _required_pure_coefficients(
        model,
        coercivity_floor,
    )
    all(isfinite, required) || return false

    for j in 1:model.D
        model.θ[pure_indices[j]] >= required[j] - atol ||
            return false
    end

    return true
end


# ---------------------------------------------------------------------------
# Polynomial evaluation
# ---------------------------------------------------------------------------

"""
    monomial(x, α)

Evaluate x^α.
"""
function monomial(
    x::AbstractVector{<:Real},
    α::AbstractVector{<:Integer},
)
    length(x) == length(α) || throw(
        DimensionMismatch("x and α must have the same length"),
    )
    return prod(x[d]^α[d] for d in eachindex(x, α))
end


function powers_for_sample(
    model::PolynomialModel,
    x::AbstractVector{<:Real},
)
    length(x) == model.D || throw(
        DimensionMismatch("sample dimension must equal model.D"),
    )
    return [monomial(x, α) for α in model.α]
end


function precompute_powers(
    x::AbstractVector{<:Real},
    L::Int,
)
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    return [
        [Float64(x[d])^degree for degree in 0:L]
        for d in eachindex(x)
    ]
end


function monomial_fast(
    α::AbstractVector{<:Integer},
    x_powers::AbstractVector{<:AbstractVector{<:Real}},
)
    length(α) == length(x_powers) || throw(
        DimensionMismatch("α and x_powers must have the same length"),
    )
    return prod(
        x_powers[d][α[d] + 1]
        for d in eachindex(α)
    )
end


"""
    precompute_monomials(model, x)

Precompute every basis monomial, first derivative, and diagonal second
derivative at x.
"""
function precompute_monomials(
    model::PolynomialModel,
    x::AbstractVector{<:Real},
)
    length(x) == model.D || throw(
        DimensionMismatch("sample dimension must equal model.D"),
    )
    all(isfinite, x) || throw(
        DomainError(x, "sample coordinates must be finite"),
    )

    K = length(model.α)
    D = model.D
    x_powers = precompute_powers(x, model.L)

    powers = zeros(Float64, K)
    first_derivatives = zeros(Float64, D, K)
    diagonal_second_derivatives = zeros(Float64, D, K)

    for k in 1:K
        α = model.α[k]
        powers[k] = monomial_fast(α, x_powers)

        for j in 1:D
            αj = α[j]

            if αj > 0
                derivative = Float64(αj)
                for d in 1:D
                    exponent = α[d] - (d == j ? 1 : 0)
                    derivative *= x_powers[d][exponent + 1]
                end
                first_derivatives[j, k] = derivative
            end

            if αj >= 2
                derivative = Float64(αj * (αj - 1))
                for d in 1:D
                    exponent = α[d] - (d == j ? 2 : 0)
                    derivative *= x_powers[d][exponent + 1]
                end
                diagonal_second_derivatives[j, k] = derivative
            end
        end
    end

    return powers, first_derivatives, diagonal_second_derivatives
end


function (model::PolynomialModel)(x::AbstractVector{<:Real})
    length(x) == model.D || throw(
        DimensionMismatch("sample dimension must equal model.D"),
    )
    return sum(
        model.θ[k] * monomial(x, model.α[k])
        for k in eachindex(model.α)
    )
end


# ---------------------------------------------------------------------------
# Energy derivatives
# ---------------------------------------------------------------------------

"""
Evaluate the first derivative of the energy using precomputed monomial
derivatives.
"""
function df_dxj(
    model::PolynomialModel,
    first_derivatives::AbstractMatrix{<:Real},
    j::Int,
)
    size(first_derivatives) == (model.D, length(model.α)) || throw(
        DimensionMismatch("first-derivative matrix has the wrong size"),
    )
    1 <= j <= model.D || throw(BoundsError(first_derivatives, (j, :)))
    return dot(model.θ, view(first_derivatives, j, :))
end


"""
Evaluate the diagonal second derivative of the energy directly at x.
"""
function d2f_dx2j(
    model::PolynomialModel,
    x::AbstractVector{<:Real},
    j::Int,
)
    length(x) == model.D || throw(
        DimensionMismatch("sample dimension must equal model.D"),
    )
    1 <= j <= model.D || throw(BoundsError(x, j))

    total = 0.0

    for (k, α) in pairs(model.α)
        αj = α[j]

        if αj >= 2
            derivative = Float64(αj * (αj - 1))
            for d in 1:model.D
                exponent = α[d] - (d == j ? 2 : 0)
                derivative *= x[d]^exponent
            end
            total += model.θ[k] * derivative
        end
    end

    return total
end


"""
Evaluate the diagonal second derivative of the energy using precomputed
monomial derivatives.
"""
function d2f_dx2j(
    model::PolynomialModel,
    diagonal_second_derivatives::AbstractMatrix{<:Real},
    j::Int,
)
    size(diagonal_second_derivatives) ==
        (model.D, length(model.α)) || throw(
        DimensionMismatch("second-derivative matrix has the wrong size"),
    )
    1 <= j <= model.D || throw(
        BoundsError(diagonal_second_derivatives, (j, :)),
    )
    return dot(
        model.θ,
        view(diagonal_second_derivatives, j, :),
    )
end


function df_dxjdθα(
    model::PolynomialModel,
    first_derivatives::AbstractMatrix{<:Real},
    k::Int,
    j::Int,
)
    size(first_derivatives) ==
        (model.D, length(model.α)) || throw(
        DimensionMismatch("first-derivative matrix has the wrong size"),
    )
    1 <= j <= size(first_derivatives, 1) || throw(
        BoundsError(first_derivatives, (j, k)),
    )
    1 <= k <= size(first_derivatives, 2) || throw(
        BoundsError(first_derivatives, (j, k)),
    )
    return first_derivatives[j, k]
end


function df_dθα(
    model::PolynomialModel,
    x::AbstractVector{<:Real},
    k::Int,
)
    1 <= k <= length(model.α) || throw(BoundsError(model.α, k))
    return monomial(x, model.α[k])
end


function hess_df_dθα(
    model::PolynomialModel,
    diagonal_second_derivatives::AbstractMatrix{<:Real},
    k::Int,
    j::Int,
)
    size(diagonal_second_derivatives) ==
        (model.D, length(model.α)) || throw(
        DimensionMismatch("second-derivative matrix has the wrong size"),
    )
    1 <= j <= size(diagonal_second_derivatives, 1) || throw(
        BoundsError(diagonal_second_derivatives, (j, k)),
    )
    1 <= k <= size(diagonal_second_derivatives, 2) || throw(
        BoundsError(diagonal_second_derivatives, (j, k)),
    )
    return diagonal_second_derivatives[j, k]
end


end # module

