module ADAM_ScoreMatching

using Random
using Printf
using Statistics
using LinearAlgebra
using Base.Threads

include("MultiDNomial.jl")
using .MultiDNomial

export adam_score_matching!,
       grad_score_matching,
       score_matching_loss,
       unnormalize_params!


# ---------------------------------------------------------------------------
# Transform a fitted energy back to the original coordinates
# ---------------------------------------------------------------------------

"""
    unnormalize_params!(model, mean_x, std_x)

If the model was fitted to z = (x - mean_x) ./ std_x, rewrite its
coefficients so that evaluating the returned polynomial at x gives the same
energy up to an additive constant.

The affine transformation preserves coercivity. The transformed constant is
set to zero because score matching cannot identify additive energy constants.
The coercivity constraint is deliberately not re-applied after this change of
coordinates: doing so would unnecessarily alter the fitted energy.
"""
function unnormalize_params!(
    model::PolynomialModel,
    mean_x::AbstractVector{<:Real},
    std_x::AbstractVector{<:Real},
)
    D = model.D
    length(mean_x) == D || throw(
        DimensionMismatch("mean_x must have length model.D"),
    )
    length(std_x) == D || throw(
        DimensionMismatch("std_x must have length model.D"),
    )
    length(model.α) == length(model.θ) || throw(
        DimensionMismatch("model.α and model.θ must have the same length"),
    )

    μ = Float64.(mean_x)
    σ = Float64.(std_x)

    all(isfinite, μ) || throw(
        DomainError(mean_x, "mean_x must contain only finite values"),
    )
    all(isfinite, σ) || throw(
        DomainError(std_x, "std_x must contain only finite values"),
    )
    all(value -> value > 0, σ) || throw(
        DomainError(std_x, "every standard deviation must be positive"),
    )
    all(isfinite, model.θ) || throw(
        DomainError(model.θ, "all polynomial coefficients must be finite"),
    )

    # Initialize every basis coefficient at zero. Keeping model.α unchanged
    # preserves its complete basis and ordering.
    accumulator = Dict{Tuple{Vararg{Int}},Float64}()
    for α in model.α
        length(α) == D || throw(
            DimensionMismatch("every multi-index must have length model.D"),
        )
        all(exponent -> exponent >= 0, α) || throw(
            ArgumentError("multi-index exponents must be nonnegative"),
        )
        sum(α) <= model.L || throw(
            ArgumentError("multi-index degree cannot exceed model.L"),
        )

        key = Tuple(α)
        haskey(accumulator, key) && throw(
            ArgumentError("model.α contains a duplicate multi-index"),
        )
        accumulator[key] = 0.0
    end

    zero_exponent = Tuple(ntuple(_ -> 0, D))

    for (coefficient, α) in zip(model.θ, model.α)
        # Partial expansion of
        # coefficient * product_d ((x_d - μ_d) / σ_d)^α_d.
        expansion = Dict{Tuple{Vararg{Int}},Float64}(
            zero_exponent => Float64(coefficient),
        )

        for d in 1:D
            αd = α[d]
            αd == 0 && continue

            next_expansion = Dict{Tuple{Vararg{Int}},Float64}()

            for (current_exponent, current_coefficient) in expansion
                for power in 0:αd
                    factor =
                        binomial(αd, power) *
                        (-μ[d])^(αd - power) /
                        σ[d]^αd

                    new_exponent = collect(current_exponent)
                    new_exponent[d] += power
                    new_key = Tuple(new_exponent)

                    next_expansion[new_key] =
                        get(next_expansion, new_key, 0.0) +
                        current_coefficient * factor
                end
            end

            expansion = next_expansion
        end

        for (key, contribution) in expansion
            haskey(accumulator, key) || error(
                "affine expansion produced a monomial outside the model basis",
            )
            accumulator[key] += contribution
        end
    end

    transformed_coefficients = [
        accumulator[Tuple(α)]
        for α in model.α
    ]
    all(isfinite, transformed_coefficients) || throw(
        OverflowError(
            "coefficient transformation overflowed; rescale the data",
        ),
    )

    model.θ .= transformed_coefficients
    zero_constant!(model)
    return model
end


# ---------------------------------------------------------------------------
# Hyvarinen objective and gradient
# ---------------------------------------------------------------------------

function _validate_precomputed(
    model::PolynomialModel,
    precomputed::AbstractVector,
)
    isempty(precomputed) && throw(
        ArgumentError("precomputed derivatives cannot be empty"),
    )

    expected_size = (model.D, length(model.α))

    for entry in precomputed
        length(entry) == 3 || throw(
            ArgumentError(
                "each precomputed entry must contain powers, first " *
                "derivatives, and diagonal second derivatives",
            ),
        )

        first_derivatives = entry[2]
        second_derivatives = entry[3]

        first_derivatives isa AbstractMatrix || throw(
            ArgumentError("precomputed first derivatives must be a matrix"),
        )
        second_derivatives isa AbstractMatrix || throw(
            ArgumentError("precomputed second derivatives must be a matrix"),
        )
        size(first_derivatives) == expected_size || throw(
            DimensionMismatch("precomputed first derivatives have the wrong size"),
        )
        size(second_derivatives) == expected_size || throw(
            DimensionMismatch("precomputed second derivatives have the wrong size"),
        )
        all(isfinite, first_derivatives) || throw(
            DomainError(
                first_derivatives,
                "precomputed first derivatives must be finite",
            ),
        )
        all(isfinite, second_derivatives) || throw(
            DomainError(
                second_derivatives,
                "precomputed second derivatives must be finite",
            ),
        )
    end

    return nothing
end

function _coordinate_weights(
    model::PolynomialModel,
    coordinate_weights,
)
    if coordinate_weights === nothing
        return ones(Float64, model.D)
    end

    length(coordinate_weights) == model.D || throw(
        DimensionMismatch("coordinate_weights must have length model.D"),
    )
    weights = Float64.(coordinate_weights)
    all(value -> isfinite(value) && value > 0, weights) || throw(
        ArgumentError(
            "coordinate_weights must contain finite positive values",
        ),
    )
    return weights
end


"""
    score_matching_loss(model, precomputed; coordinate_weights=nothing)

Evaluate

    (1/N) sum_i sum_j [
        w_j * (
            0.5 * (partial_j f(x_i))^2 - partial_jj f(x_i)
        )
    ].

The default weights are all one. When the derivatives are evaluated in
standardized coordinates z_j = (x_j - mean_j) / std_j, use
w_j = 1 / std_j^2 to recover the original-coordinate objective.

The Laplacian term is linear in the fitted coefficients:

    partial_jj f(x_i)
        = sum_k theta_k * partial_jj phi_k(x_i).
"""
function score_matching_loss(
    model::PolynomialModel,
    precomputed::AbstractVector;
    coordinate_weights=nothing,
    validate_precomputed::Bool=true,
)
    validate_precomputed &&
        _validate_precomputed(model, precomputed)
    weights = _coordinate_weights(model, coordinate_weights)

    N = length(precomputed)
    worker_count = min(Threads.nthreads(), N)
    partial_losses = zeros(Float64, worker_count)

    # Each logical worker writes to its own accumulator. This remains safe even
    # if Julia migrates a task between physical threads.
    Threads.@threads for worker in 1:worker_count
        local_loss = 0.0

        for i in worker:worker_count:N
            _, first_derivatives, second_derivatives = precomputed[i]

            for j in 1:model.D
                score_component = dot(
                    model.θ,
                    view(first_derivatives, j, :),
                )
                laplacian_component = dot(
                    model.θ,
                    view(second_derivatives, j, :),
                )

                local_loss += weights[j] * (
                    0.5 * score_component^2 -
                    laplacian_component
                )
            end
        end

        partial_losses[worker] = local_loss
    end

    loss = sum(partial_losses) / N
    isfinite(loss) || throw(
        DomainError(loss, "the score-matching loss is not finite"),
    )
    return loss
end


"""
    grad_score_matching(model, precomputed; coordinate_weights=nothing)

Evaluate the gradient of score_matching_loss from precomputed monomial
derivatives. coordinate_weights must match the weights used in the loss.
"""
function grad_score_matching(
    model::PolynomialModel,
    precomputed::AbstractVector;
    coordinate_weights=nothing,
    validate_precomputed::Bool=true,
)
    validate_precomputed &&
        _validate_precomputed(model, precomputed)
    weights = _coordinate_weights(model, coordinate_weights)

    N = length(precomputed)
    K = length(model.α)
    worker_count = min(Threads.nthreads(), N)
    partial_gradients = [
        zeros(Float64, K)
        for _ in 1:worker_count
    ]

    Threads.@threads for worker in 1:worker_count
        local_gradient = partial_gradients[worker]

        for i in worker:worker_count:N
            _, first_derivatives, second_derivatives = precomputed[i]

            for j in 1:model.D
                score_component = dot(
                    model.θ,
                    view(first_derivatives, j, :),
                )

                for k in 1:K
                    local_gradient[k] +=
                        weights[j] * (
                            score_component * first_derivatives[j, k] -
                            second_derivatives[j, k]
                        )
                end
            end
        end
    end

    gradient = zeros(Float64, K)
    for partial_gradient in partial_gradients
        gradient .+= partial_gradient
    end
    gradient ./= N

    # The constant coefficient is fixed at zero and should accumulate neither
    # a gradient nor Adam momentum.
    fixed_mask = fixed_coefficient_mask(model)
    for k in eachindex(gradient)
        fixed_mask[k] && (gradient[k] = 0.0)
    end

    all(isfinite, gradient) || throw(
        DomainError(gradient, "the score-matching gradient is not finite"),
    )
    return gradient
end


"""
    grad_score_matching(model, samples, precomputed)

Backward-compatible method for the original three-argument API. The samples
are used only to verify the number of precomputed entries.
"""
function grad_score_matching(
    model::PolynomialModel,
    samples::AbstractVector,
    precomputed::AbstractVector;
    coordinate_weights=nothing,
    validate_precomputed::Bool=true,
)
    length(samples) == length(precomputed) || throw(
        DimensionMismatch(
            "samples and precomputed derivatives must have equal length",
        ),
    )
    return grad_score_matching(
        model,
        precomputed;
        coordinate_weights=coordinate_weights,
        validate_precomputed=validate_precomputed,
    )
end


# ---------------------------------------------------------------------------
# Adam update
# ---------------------------------------------------------------------------

mutable struct AdamParam
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end


function _validate_adam_hyperparameters(
    η::Real,
    β1::Real,
    β2::Real,
    ϵ::Real,
    clip_grad::Real,
)
    isfinite(η) && η > 0 || throw(
        ArgumentError("η must be finite and positive"),
    )
    isfinite(β1) && 0 <= β1 < 1 || throw(
        ArgumentError("β1 must satisfy 0 <= β1 < 1"),
    )
    isfinite(β2) && 0 <= β2 < 1 || throw(
        ArgumentError("β2 must satisfy 0 <= β2 < 1"),
    )
    isfinite(ϵ) && ϵ > 0 || throw(
        ArgumentError("ϵ must be finite and positive"),
    )
    (
        (isfinite(clip_grad) && clip_grad > 0) ||
        clip_grad == Inf
    ) || throw(
        ArgumentError("clip_grad must be positive or Inf"),
    )
    return nothing
end


function adam_step!(
    θ::Vector{Float64},
    gradient::AbstractVector{<:Real},
    state::AdamParam;
    η::Real=0.01,
    β1::Real=0.9,
    β2::Real=0.999,
    ϵ::Real=1.0e-8,
    clip_grad::Real=Inf,
)
    _validate_adam_hyperparameters(η, β1, β2, ϵ, clip_grad)

    length(gradient) == length(θ) || throw(
        DimensionMismatch("gradient and parameter vectors must have equal length"),
    )
    length(state.m) == length(θ) || throw(
        DimensionMismatch("Adam first moment has the wrong length"),
    )
    length(state.v) == length(θ) || throw(
        DimensionMismatch("Adam second moment has the wrong length"),
    )
    all(isfinite, gradient) || throw(
        DomainError(gradient, "gradient must contain only finite values"),
    )

    effective_gradient = Float64.(gradient)

    if clip_grad < Inf
        gradient_norm = norm(effective_gradient)
        if gradient_norm > clip_grad
            effective_gradient .*= clip_grad / gradient_norm
        end
    end

    state.t += 1
    state.m .=
        β1 .* state.m .+
        (1 - β1) .* effective_gradient
    state.v .=
        β2 .* state.v .+
        (1 - β2) .* (effective_gradient .^ 2)

    first_moment_hat = state.m ./ (1 - β1^state.t)
    second_moment_hat = state.v ./ (1 - β2^state.t)

    θ .-=
        η .* first_moment_hat ./
        (sqrt.(second_moment_hat) .+ ϵ)

    all(isfinite, θ) || throw(
        DomainError(θ, "Adam produced non-finite parameters"),
    )
    return θ
end


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

function _validated_sample_matrix(
    model::PolynomialModel,
    samples::AbstractVector{<:AbstractVector{<:Real}},
)
    N = length(samples)
    N >= 2 || throw(
        ArgumentError("score matching requires at least two observations"),
    )

    matrix = Matrix{Float64}(undef, N, model.D)

    for i in 1:N
        length(samples[i]) == model.D || throw(
            DimensionMismatch(
                "sample $i has dimension $(length(samples[i])); " *
                "expected $(model.D)",
            ),
        )
        all(isfinite, samples[i]) || throw(
            DomainError(samples[i], "sample $i contains a non-finite value"),
        )
        matrix[i, :] .= samples[i]
    end

    return matrix
end


function _cosine_step_size(
    epoch::Int,
    max_epochs::Int,
    initial_step_size::Real,
)
    # The hypothetical epoch max_epochs + 1 reaches zero. Consequently, every
    # epoch that actually runs has a strictly positive step size.
    phase = (epoch - 1) / max_epochs
    return Float64(initial_step_size) *
           0.5 *
           (1 + cos(π * phase))
end


"""
    adam_score_matching!(model, samples; kwargs...)

Fit a polynomial energy by mini-batch score matching. Training occurs in
standardized coordinates, with inverse-variance coordinate weights so that
the optimized criterion is the original-coordinate Hyvarinen objective.
After every Adam step, the constant and coercivity constraints supplied by
MultiDNomial are enforced.
"""
function adam_score_matching!(
    model::PolynomialModel,
    samples::AbstractVector{<:AbstractVector{<:Real}};
    η::Real=0.01,
    β1::Real=0.9,
    β2::Real=0.999,
    ϵ::Real=1.0e-8,
    tol_loss::Real=1.0e-6,
    batch_size::Int=1000,
    max_epochs::Int=5000,
    check_every::Int=50,
    verbose::Bool=true,
    clip_grad::Real=Inf,
    coercivity_floor::Real=MultiDNomial.DEFAULT_COERCIVITY_FLOOR,
    min_std::Real=sqrt(eps(Float64)),
    rng::AbstractRNG=Random.default_rng(),
)
    _validate_adam_hyperparameters(η, β1, β2, ϵ, clip_grad)
    isfinite(tol_loss) && tol_loss >= 0 || throw(
        ArgumentError("tol_loss must be finite and nonnegative"),
    )
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    max_epochs > 0 || throw(ArgumentError("max_epochs must be positive"))
    check_every > 0 || throw(ArgumentError("check_every must be positive"))
    isfinite(min_std) && min_std > 0 || throw(
        ArgumentError("min_std must be finite and positive"),
    )

    data = _validated_sample_matrix(model, samples)
    N = size(data, 1)

    # Validate the model and put its standardized-coordinate coefficients in
    # the feasible set before doing any expensive precomputation.
    enforce_energy_constraints!(
        model;
        coercivity_floor=coercivity_floor,
    )

    mean_x = vec(mean(data; dims=1))
    std_x = vec(std(data; dims=1, corrected=true))

    bad_coordinates = findall(
        value -> !isfinite(value) || value <= min_std,
        std_x,
    )
    isempty(bad_coordinates) || throw(
        ArgumentError(
            "coordinates $(bad_coordinates) have zero or near-zero " *
            "empirical standard deviation",
        ),
    )

    standardized_samples = [
        [
            (data[i, j] - mean_x[j]) / std_x[j]
            for j in 1:model.D
        ]
        for i in 1:N
    ]
    coordinate_weights = 1.0 ./ (std_x .^ 2)

    verbose && println("Precomputing polynomial derivatives...")
    precomputed = [
        precompute_monomials(model, sample)
        for sample in standardized_samples
    ]
    _validate_precomputed(model, precomputed)

    fixed_mask = fixed_coefficient_mask(model)

    K = length(model.α)
    state = AdamParam(zeros(K), zeros(K), 0)

    previous_loss = nothing
    converged = false
    final_epoch = 0

    for epoch in 1:max_epochs
        final_epoch = epoch
        shuffled_indices = randperm(rng, N)
        step_size = _cosine_step_size(epoch, max_epochs, η)

        for batch_start in 1:batch_size:N
            batch_end = min(batch_start + batch_size - 1, N)
            batch_indices = shuffled_indices[batch_start:batch_end]
            batch_precomputed = precomputed[batch_indices]

            gradient = grad_score_matching(
                model,
                batch_precomputed;
                coordinate_weights=coordinate_weights,
                validate_precomputed=false,
            )
            adam_step!(
                model.θ,
                gradient,
                state;
                η=step_size,
                β1=β1,
                β2=β2,
                ϵ=ϵ,
                clip_grad=clip_grad,
            )

            enforce_energy_constraints!(
                model;
                coercivity_floor=coercivity_floor,
            )

            # Prevent any fixed coefficient from retaining optimizer state.
            for k in eachindex(fixed_mask)
                if fixed_mask[k]
                    state.m[k] = 0.0
                    state.v[k] = 0.0
                end
            end
        end

        should_check =
            epoch == 1 ||
            epoch % check_every == 0 ||
            epoch == max_epochs

        if should_check
            current_loss = score_matching_loss(
                model,
                precomputed;
                coordinate_weights=coordinate_weights,
                validate_precomputed=false,
            )

            if previous_loss === nothing
                if verbose
                    @printf(
                        "Epoch %d, score-matching loss = %.8e\n",
                        epoch,
                        current_loss,
                    )
                end
            else
                denominator = max(
                    abs(previous_loss),
                    abs(current_loss),
                    eps(Float64),
                )
                relative_change =
                    abs(current_loss - previous_loss) /
                    denominator

                if verbose
                    @printf(
                        "Epoch %d, score-matching loss = %.8e, relative change = %.3e\n",
                        epoch,
                        current_loss,
                        relative_change,
                    )
                end

                if relative_change < tol_loss
                    converged = true
                    verbose && println(
                        "Converged at epoch $epoch: relative loss " *
                        "change < $tol_loss",
                    )
                    break
                end
            end

            previous_loss = current_loss
        end
    end

    verbose && !converged && println(
        "Stopped after $final_epoch epochs without satisfying " *
        "the loss-change tolerance.",
    )

    # The model is coercive in standardized coordinates. An invertible affine
    # transformation preserves coercivity, so no second projection is needed.
    unnormalize_params!(model, mean_x, std_x)

    verbose && println("Final coefficients: ", model.θ)
    return model
end


end # module

