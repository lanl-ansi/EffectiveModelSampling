using Distributions
using DataFrames
using Plots
using Random
using Statistics

include("../IMPORTABLES/ImportanceSampling.jl")
using .ImportanceSampling

Random.seed!(1234)


# ------------------------------------------------------------------
# Shared settings
# ------------------------------------------------------------------

theta_star = 1 / 4.0
p_star = Normal(0.0, sqrt(1 / theta_star))

delta_total = 0.05
delta_IS = delta_total / 2
delta_SM = delta_total / 2

R = 500

epsilon_vals = range(
    0.05,
    1.0,
    length=30,
)

M_vals = unique(
    round.(Int, 10 .^ range(1, 6, length=100))
)

n_vals = unique(
    round.(Int, 10 .^ range(1, 6, length=100))
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

function scoreMatchingTrial(
    M::Int,
    theta_star::Float64,
)
    # theta_star * sum(X_i^2) ~ Chisq(M)
    Y = rand(Chisq(M))

    inverse_theta_SM =
        Y / (M * theta_star)

    theta_SM_trial =
        1 / inverse_theta_SM

    error_SM =
        abs(inverse_theta_SM - 1 / theta_star)

    return error_SM, theta_SM_trial
end


function fitPrecision(M::Int, distribution)
    samples = rand(distribution, M)

    return M / sum(abs2, samples)
end


function C_kappa(kappa::Float64)
    kappa > 0 || throw(
        DomainError(kappa, "The bound requires kappa > 0")
    )

    return (
        3 * (kappa + 1)^3 /
        (8 * kappa^(5 / 2)) - 1
    )
end


# ==================================================================
# EXPERIMENT A: score-matching sample complexity M*(epsilon)
# ==================================================================

df_M = DataFrame(
    epsilon=Float64[],
    delta=Float64[],
    M_estimate=Union{Missing, Int}[],
    M_theory=Int[],
)


for epsilon in epsilon_vals
    tolerance = epsilon / theta_star
    M_estimate = missing

    for M in M_vals
        successes = 0

        for _ in 1:R
            # Do not assign the returned parameter to theta_SM.
            error_SM, _ =
                scoreMatchingTrial(M, theta_star)

            if error_SM <= tolerance
                successes += 1
            end
        end

        if successes == R
            M_estimate = M
            break
        end
    end

    # Second-case result:
    #
    # |1/theta_SM - 1/theta_star|
    # <= (1/theta_star) *
    #    sqrt(8/M * log(2/delta_SM))
    #
    # Setting the RHS <= epsilon/theta_star gives:
    #
    # M >= 8 log(2/delta_SM) / epsilon^2.
    M_theory = ceil(
        Int,
        8 * log(2 / delta_SM) / epsilon^2,
    )

    push!(
        df_M,
        (
            epsilon=epsilon,
            delta=delta_SM,
            M_estimate=M_estimate,
            M_theory=M_theory,
        ),
    )

    println(
        "A: epsilon=$epsilon, " *
        "M_estimate=$M_estimate, " *
        "M_theory=$M_theory"
    )
end


# ==================================================================
# EXPERIMENT B: importance-sampling sample complexity n*(epsilon)
# ==================================================================

# These fits are held fixed throughout Experiment B.
# M_fit is only used to construct the two fitted distributions.
M_fit = 10_000

theta_SM_fit =
    fitPrecision(M_fit, p_star)

theta_G_fit =
    fitPrecision(M_fit, p_star)

sigma_squared_fit =
    1 / theta_G_fit

kappa_fit =
    2 * theta_SM_fit * sigma_squared_fit - 1

kappa_fit > 0 || error(
    "The fitted proposal produced kappa=$kappa_fit. " *
    "The importance-sampling bound requires kappa > 0."
)

p_SM_dist =
    Normal(0.0, sqrt(1 / theta_SM_fit))

p_G_fit =
    Normal(0.0, sqrt(sigma_squared_fit))

# ImportanceSampling expects its target as a function.
p_SM(x) = pdf(p_SM_dist, x)

h(x) = x^2

target_IS =
    1 / theta_SM_fit

C_fit =
    C_kappa(kappa_fit)


df_n = DataFrame(
    epsilon=Float64[],
    delta=Float64[],
    kappa=Float64[],
    n_estimate=Union{Missing, Int}[],
    n_theory=Float64[],
)


for epsilon in epsilon_vals
    tolerance =
        epsilon / theta_SM_fit

    n_estimate = missing

    for n in n_vals
        successes = 0

        # theta_SM_fit, theta_G_fit and kappa_fit remain fixed.
        # Only the importance samples are regenerated.
        for _ in 1:R
            I_n =
                ImportanceSampling.importanceSampling(
                    n,
                    p_SM,
                    p_G_fit,
                    h;
                    verbose=false,
                )

            error_IS =
                abs(I_n - target_IS)

            if error_IS <= tolerance
                successes += 1
            end
        end

        if successes == R
            n_estimate = n
            break
        end
    end

    # First-case result:
    #
    # |I_n - 1/theta_SM|
    # <= (1/theta_SM) *
    #    sqrt(C(kappa)/(delta_IS*n))
    #
    # Setting the RHS <= epsilon/theta_SM gives:
    #
    # n >= C(kappa)/(delta_IS*epsilon^2).
    n_theory =
        C_fit / (delta_IS * epsilon^2)

    push!(
        df_n,
        (
            epsilon=epsilon,
            delta=delta_IS,
            kappa=kappa_fit,
            n_estimate=n_estimate,
            n_theory=n_theory,
        ),
    )

    println(
        "B: epsilon=$epsilon, " *
        "n_estimate=$n_estimate, " *
        "n_theory=$n_theory"
    )
end


# ==================================================================
# EXPERIMENT C: direct verification of the complete bound
#
# A relationship between M and n must be chosen. Here M=n=s.
# ==================================================================

function fullBoundTrial(
    M::Int,
    n::Int,
    theta_star::Float64,
    p_star,
    delta_IS::Float64,
    delta_SM::Float64,
)
    theta_SM =
        fitPrecision(M, p_star)

    proposal_attempts = 0
    theta_G = NaN
    sigma_squared = NaN
    kappa = -Inf

    while kappa <= 0
        proposal_attempts += 1

        theta_G =
            fitPrecision(M, p_star)

        sigma_squared =
            1 / theta_G

        kappa =
            2 * theta_SM * sigma_squared - 1
    end

    p_SM_trial_dist =
        Normal(0.0, sqrt(1 / theta_SM))

    p_G_trial =
        Normal(0.0, sqrt(sigma_squared))

    p_SM_trial =
        x -> pdf(p_SM_trial_dist, x)

    I_n =
        ImportanceSampling.importanceSampling(
            n,
            p_SM_trial,
            p_G_trial,
            x -> x^2;
            verbose=false,
        )

    C_trial =
        C_kappa(kappa)

    first_bound =
        (1 / theta_SM) *
        sqrt(
            C_trial /
            (delta_IS * n)
        )

    second_bound =
        (1 / theta_star) *
        sqrt(
            8 / M *
            log(2 / delta_SM)
        )

    total_bound =
        first_bound + second_bound

    total_error =
        abs(I_n - 1 / theta_star)

    return (
        total_error=total_error,
        first_bound=first_bound,
        second_bound=second_bound,
        total_bound=total_bound,
        success=total_error <= total_bound,
        kappa=kappa,
        proposal_attempts=proposal_attempts,
    )
end

# Linear grid because the final plot uses linear axes.
s_vals = unique(
    round.(Int, range(100, 50_000, length=30))
)

df_full = DataFrame(
    s=Int[],
    trial=Int[],
    total_error=Float64[],
    first_bound=Float64[],
    second_bound=Float64[],
    total_bound=Float64[],
    success=Bool[],
    kappa=Float64[],
    proposal_attempts=Int[],
)


for s in s_vals
    successes = 0

    for trial in 1:R
        result =
            fullBoundTrial(
                s,              # M
                s,              # n
                theta_star,
                p_star,
                delta_IS,
                delta_SM,
            )

        successes += result.success

        push!(
            df_full,
            (
                s=s,
                trial=trial,
                total_error=result.total_error,
                first_bound=result.first_bound,
                second_bound=result.second_bound,
                total_bound=result.total_bound,
                success=result.success,
                kappa=result.kappa,
                proposal_attempts=
                    result.proposal_attempts,
            ),
        )
    end

    println(
        "Full bound: M=n=$s, " *
        "successes=$successes/$R"
    )
end


df_full_summary = combine(
    groupby(df_full, :s),
    :total_error => median => :median_error,
    :total_bound => median => :median_bound,
    :success => mean => :coverage,
)


# ==================================================================
# Plots: linear axes
# ==================================================================

df_M_empirical =
    dropmissing(df_M, :M_estimate)

plt_M = scatter(
    df_M_empirical.epsilon,
    df_M_empirical.M_estimate;
    marker=:circle,
    markersize=6,
    label="Empirical M*(epsilon)",
    xlabel="Relative accuracy epsilon",
    ylabel="Required sample size M",
    title="Score-matching sample complexity",
    size=(900, 600),
    dpi=300,
)

plot!(
    plt_M,
    df_M.epsilon,
    df_M.M_theory;
    linewidth=2,
    label="Theory: 8 log(2/delta_SM) / epsilon^2",
)

savefig(
    plt_M,
    "sample_complexity_M.png",
)


df_n_empirical =
    dropmissing(df_n, :n_estimate)

plt_n = scatter(
    df_n_empirical.epsilon,
    df_n_empirical.n_estimate;
    marker=:circle,
    markersize=6,
    label="Empirical n*(epsilon)",
    xlabel="Relative accuracy epsilon",
    ylabel="Required sample size n",
    title="Importance-sampling sample complexity",
    size=(900, 600),
    dpi=300,
)

plot!(
    plt_n,
    df_n.epsilon,
    df_n.n_theory;
    linewidth=2,
    label="Theory: C(kappa) / (delta_IS epsilon^2)",
)

savefig(
    plt_n,
    "sample_complexity_n.png",
)


plt_full_error = scatter(
    df_full.s,
    df_full.total_error;
    markersize=3,
    alpha=0.30,
    label="Roundwise total error",
    xlabel="Common sample size s (M=n=s)",
    ylabel="Error or bound",
    title="Complete bound",
)

scatter!(
    plt_full_error,
    df_full.s,
    df_full.total_bound;
    markersize=3,
    alpha=0.30,
    label="Roundwise complete bound",
)

plot!(
    plt_full_error,
    df_full_summary.s,
    df_full_summary.median_error;
    linewidth=3,
    label="Median error",
)

plot!(
    plt_full_error,
    df_full_summary.s,
    df_full_summary.median_bound;
    linewidth=3,
    label="Median bound",
)


plt_coverage = plot(
    df_full_summary.s,
    df_full_summary.coverage;
    marker=:circle,
    linewidth=2,
    label="Empirical coverage",
    xlabel="Common sample size s (M=n=s)",
    ylabel="Fraction satisfying bound",
    ylims=(0.0, 1.05),
)

hline!(
    plt_coverage,
    [1 - delta_total];
    linestyle=:dash,
    linewidth=2,
    label="Claimed coverage: 1-delta",
)


plt_full = plot(
    plt_full_error,
    plt_coverage;
    layout=(2, 1),
    size=(900, 900),
    dpi=300,
)

savefig(
    plt_full,
    "sample_complexity_full.png",
)
