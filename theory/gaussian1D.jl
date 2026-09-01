using Distributions
using DataFrames
using Plots
include("../IMPORTABLES/ImportanceSampling.jl")
using .ImportanceSampling
using Random
Random.seed!(1234)

# One independent score-matching trial.
function scoreMatchingTrial(M::Int, theta_star::Float64)
    # From the proof:
    # Y = theta_star * sum(X_i^2) ~ Chisq(M)
    Y = rand(Chisq(M))

    # 1/theta_SM = (1/M)sum(X_i^2) = Y/(M*theta_star)
    inverse_theta_SM = Y / (M * theta_star)
    theta_SM = 1 / inverse_theta_SM
    error_SM = abs(inverse_theta_SM - 1/theta_star)

    return error_SM, theta_SM
end

theta_star = 1 / 4.0
delta = 0.05
R = 20

M_vals = unique(
    round.(Int, 10 .^ range(1, 9, length=100))
)

epsilon_vals = 10 .^ range(
    log10(0.05),
    log10(1.0),
    length=20,
)


# FIRST PART ---------------------------------------------------
df = DataFrame(
    epsilon = Float64[],
    delta = Float64[],
    M_estimate = Union{Missing, Int}[],
    M_theory = Int[],
)


for epsilon in epsilon_vals
    tolerance = epsilon / theta_star
    M_estimate = missing

    # Search candidate values of M from smallest to largest.
    for M in M_vals
        successes = 0

        # R independent trials at this same M.
        for round in 1:R
            error_SM, theta_SM =
                scoreMatchingTrial(M, theta_star)

            if error_SM <= tolerance
                successes += 1
            end
        end

        println(
            "epsilon=$epsilon, M=$M, successes=$successes/$R"
        )

        # Lokhov-style crossing criterion.
        if successes == R
            M_estimate = M
            break
        end
    end

    # Isolated score-matching result:
    #
    # P(error_SM >= epsilon/theta_star)
    #     <= 2 exp(-M epsilon^2 / 8).
    #
    # Therefore:
    # M >= 8 log(2/delta) / epsilon^2.
    M_theory = ceil(
        Int,
        8 * log(2 / delta) / epsilon^2,
    )

    push!(
        df,
        (
            epsilon = epsilon,
            delta = delta,
            M_estimate = M_estimate,
            M_theory = M_theory,
        ),
    )
end

plt = scatter(
    df.epsilon,
    df.M_estimate,
    marker = :circle,
    markersize = 6,
    label = "Empirical M*(epsilon)",
    xscale = :log10,
    yscale = :log10,
    xlabel = "Relative accuracy epsilon",
    ylabel = "Required sample size M",
    title = "Score-matching sample complexity",
    size = (900, 600),
    dpi = 300,
)

plot!(
    plt,
    df.epsilon,
    df.M_theory,
    linewidth = 2,
    label = "Theory: M proportional to epsilon^(-2)",
)

savefig("sample_complexity_M.png")


# SECOND PART ---------------------------------------------------

n_vals = unique(
    round.(Int, 10 .^ range(1, 6, length=100))
)

M_fit = 10_000

theta_star = 1 / 4.0
p_star = Normal(
    0.0,
    sqrt(1 / theta_star),
)

# Fit theta_SM from M target samples.
samples_SM = rand(p_star, M_fit)

theta_SM = M_fit / sum(samples_SM .^ 2)
target_IS = 1 / theta_SM

# Independently fit theta_G from M target samples.
samples_G = rand(p_star, M_fit)

theta_G = M_fit / sum(samples_G .^ 2)

# Proposal variance obtained from its fitted precision.
sigma_squared = 1 / theta_G

# Equivalent expression:
# kappa = 2*theta_SM/theta_G - 1
kappa = 2 * theta_SM * sigma_squared - 1

if kappa <= 0
    error(
        "The fitted proposal does not satisfy kappa > 0. " *
        "The variance bound is not applicable."
    )
end

p_G = Normal(0.0,sqrt(sigma_squared))

# Learned target.
q_SM(x) =exp(-theta_SM * x^2 / 2)
Z_SM =sqrt(2*π / theta_SM)
p_SM(x) = exp(-theta_SM * x^2 / 2) / Z_SM


h(x) = x^2

df_n = DataFrame(
    epsilon=Float64[],
    delta=Float64[],
    kappa=Float64[],
    n_estimate=Union{Missing, Int}[],
    n_theory=Float64[],
)

for epsilon in epsilon_vals
    tolerance = epsilon / theta_SM

    n_estimate = missing

    for n in n_vals
        successes = 0

        # Only the importance samples are regenerated here.
        # The fitted theta_SM and theta_G remain fixed.
        for trial in 1:R
            I_n =
                ImportanceSampling.importanceSampling(
                    n,
                    p_SM,
                    p_G,
                    h;
                    verbose=false,
                )

            error_IS =
                abs(I_n - 1/theta_SM)

            if error_IS <= tolerance
                successes += 1
            end
        end

        println(
            "epsilon=$epsilon, n=$n, " *
            "successes=$successes/$R"
        )

        if successes == R
            n_estimate = n
            break
        end
    end

    C_kappa = 3 * (kappa + 1)^3 / (8 * kappa^(5/2)) - 1

    n_theory = C_kappa / (delta * epsilon^2)

    push!(
        df_n,
        (
            epsilon=epsilon,
            delta=delta,
            kappa=kappa,
            n_estimate=n_estimate,
            n_theory=n_theory,
        ),
    )
end




plt_n = scatter(
    df_n.epsilon,
    df_n.n_estimate,
    marker=:circle,
    markersize=6,
    label="Empirical n*(epsilon)",
    xscale=:log10,
    yscale=:log10,
    xlabel="Relative accuracy epsilon",
    ylabel="Required sample size n",
    title="Importance-sampling sample complexity",
    size=(900, 600),
    dpi=300,
)

plot!(
    plt_n,
    df_n.epsilon,
    df_n.n_theory,
    linewidth=2,
    label="Theory: n proportional to epsilon^(-2)",
)

savefig(
    plt_n,
    "sample_complexity_n.png",
)

