function sample_dynamichmc(prob, ndraws, nchains=4; kwargs...)
    idata = _sample_dynamichmc(prob, ndraws, nchains; kwargs...)
    return idata
end

function _sample_dynamichmc(
    prob,
    ndraws,
    nchains;
    rng=Random.default_rng(),
    reporter=DynamicHMC.NoProgressReport(),
    executor=Transducers.PreferParallel(),
    kwargs...,
)
    trans = Transducers.Map() do seed
        rng_chain = deepcopy(rng)
        Random.seed!(rng_chain, seed)
        count_prob = EvalCountingProblem(prob)
        sample = DynamicHMC.mcmc_with_warmup(
            rng_chain, count_prob, ndraws; reporter, kwargs...
        )
        return sample, count_prob.num_evals, count_prob.num_grad_evals
    end
    # generate seeds to be used for each chain. with TaskLocalRNG this isn't necessary, but
    # it does guarantee results are the same regardless of the `rng` or `executor` we use.
    seeds = rand(rng, UInt, nchains)
    results = Folds.collect(trans(seeds), executor)
    samples = first.(results)
    # get the number of function evaluations and gradient evaluations
    num_evals = getindex.(results, 2)
    num_grad_evals = getindex.(results, 3)
    postmat = DynamicHMC.stack_posterior_matrices(samples)
    metadata = Dict("num_evals" => num_evals, "num_grad_evals" => num_grad_evals)
    posterior = InferenceObjects.convert_to_dataset(postmat; attrs=metadata)
    # TODO: add counts to sample_stats?
    # TODO: add wallclock time
    sample_stats = InferenceObjects.convert_to_dataset(map(_dynamichmc_stats, samples))
    return InferenceObjects.InferenceData(; posterior, sample_stats)
end

function _dynamichmc_stats(result)
    step_size = result.ϵ
    # TODO: remove the draw dimension from the inverse metric
    inv_metric = result.κ.M⁻¹
    return map(result.tree_statistics) do stat
        termination = stat.termination
        return (
            energy=stat.π,
            tree_depth=stat.depth,
            acceptance_rate=stat.acceptance_rate,
            n_steps=stat.steps,
            diverging=termination.left == termination.right,
            turning=termination.left < termination.right,
            step_size,
            inv_metric,
        )
    end
end

# wrapper to count the number of function evaluations and gradient evaluations
mutable struct EvalCountingProblem{P}
    const prob::P
    num_evals::Int
    num_grad_evals::Int
end
EvalCountingProblem(prob) = EvalCountingProblem(prob, 0, 0)

function LogDensityProblems.capabilities(::Type{<:EvalCountingProblem{P}}) where {P}
    return LogDensityProblems.capabilities(P)
end

function LogDensityProblems.dimension(prob::EvalCountingProblem)
    return LogDensityProblems.dimension(prob.prob)
end

function LogDensityProblems.logdensity(prob::EvalCountingProblem, x)
    prob.num_evals += 1
    return LogDensityProblems.logdensity(prob.prob, x)
end

function LogDensityProblems.logdensity_and_gradient(prob::EvalCountingProblem, x)
    prob.num_grad_evals += 1
    return LogDensityProblems.logdensity_and_gradient(prob.prob, x)
end
