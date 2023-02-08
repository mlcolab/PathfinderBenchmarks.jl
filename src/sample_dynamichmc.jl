"""
    sample_dynamichmc(prob, ndraws, nchains; kwargs...) -> InferenceObjects.InferenceData

Sample the LogDensityProblem `prob` using DynamicHMC.jl.

# Keywords

- `rng::AbstractRNG`: random number generator
- `executor=Transducers.PreferParallel()`: executor used to sample the chains in parallel
- `ntries::Int=100`
- `initializations=fill((), nchains)`: vector of initializations for each chain. Each entry
    should be the same as the `initialization` keyword argument to
    `DynamicHMC.mcmc_with_warmup`.
- `dims=(;)`: map from parameter names to dimension names, forwarded to
    `InferenceObjects.convert_to_dataset`.
- `coords=(;)`: map from dimension names in `dims` to named indices, forwarded to
    `InferenceObjects.convert_to_dataset`.
- Remaining keywords are forwarded to `DynamicHMC.mcmc_with_warmup`.
"""
sample_dynamichmc
function sample_dynamichmc(
    prob::StanLogDensityProblems.StanProblem, ndraws::Int, nchains::Int; kwargs...
)
    param_unc_names = BridgeStan.param_unc_names(prob.model)
    # sample unconstrained parameters
    idata = _sample_dynamichmc(
        prob,
        ndraws,
        nchains;
        dims=(; x=[:param_unc]),
        coords=(; param_unc=param_unc_names),
        kwargs...,
    )
    # constrain parameters
    x = _constrain_params(prob, idata.posterior.x)
    # replace unconstrained parameters with constrained parameters
    posterior = merge(idata.posterior, (; x))
    return merge(idata, InferenceObjects.InferenceData(; posterior))
end

function _constrain_params(prob, x)
    model = prob.model
    param_names = BridgeStan.param_names(model)
    xnew = mapslices(Base.Fix1(BridgeStan.param_constrain, model), x; dims=:param_unc)
    param_dim = DimensionalData.Dim{:param}(param_names)
    return DimensionalData.set(xnew, :param_unc => param_dim)
end

function _sample_dynamichmc(
    prob,
    ndraws,
    nchains;
    rng=Random.default_rng(),
    reporter=DynamicHMC.NoProgressReport(),
    initializations=fill((), nchains),
    executor=Transducers.PreferParallel(),
    ntries=100,
    dims=(;),
    coords=(;),
    warmup_stages=DynamicHMC.default_warmup_stages(),
    kwargs...,
)
    # transducer for sampling, supports multiple parallelism approaches
    trans = Transducers.MapSplat() do seed, initialization
        rng_chain = deepcopy(rng)
        Random.seed!(rng_chain, seed)
        # avoid including compilation time in the timing

        count_prob_warmup = EvalCountingProblem(prob)
        (warmup_initialization, ntries_warmup), time_warmup = @timed _dhmc_warmup_until_succeeds(
            rng_chain,
            count_prob_warmup,
            ntries;
            reporter,
            initialization,
            warmup_stages,
            kwargs...,
        )
        nevals_warmup = num_evaluations(count_prob_warmup)

        count_prob_sample = EvalCountingProblem(prob)
        sample, time_sample = @timed dhmc_sample(
            rng_chain,
            count_prob_sample,
            ndraws;
            reporter,
            initialization=warmup_initialization,
            kwargs...,
        )
        nevals_sample = num_evaluations(count_prob_sample)

        return (;
            sample,
            num_evals=vcat(nevals_warmup', nevals_sample'),
            time=[time_warmup, time_sample],
            num_tries=ntries_warmup,
        )
    end

    # generate seeds to be used for each chain. with TaskLocalRNG this isn't necessary, but
    # it does guarantee results are the same regardless of the `rng` or `executor` we use.
    seeds = rand(rng, UInt, nchains)
    iter = Transducers.withprogress(zip(seeds, initializations); interval=1e-3)
    results = Folds.collect(trans(iter), executor)
    samples = map(first, results)

    # get posterior draws
    postmat = DynamicHMC.stack_posterior_matrices(samples)
    posterior = InferenceObjects.convert_to_dataset(postmat; dims, coords)

    # get statistics
    stats = map(_dynamichmc_stats, samples)
    draw_stats = map(first, stats)
    chain_stats = map(Base.vect ∘ last, stats)
    summary_stats = map(Base.vect ∘ Base.tail, results)
    dims_summary = merge((; num_evals=[:stage, :eval_type], time=[:stage]), dims)
    coords_summary = merge(
        (; eval_type=["fun", "grad", "hess"], stage=["warmup", "sampling"]), coords
    )
    sample_stats = merge(
        InferenceObjects.convert_to_dataset(draw_stats; dims, coords),
        dropdims(
            InferenceObjects.convert_to_dataset(chain_stats; dims, coords); dims=:draw
        ),
        dropdims(
            InferenceObjects.convert_to_dataset(
                summary_stats; dims=dims_summary, coords=coords_summary
            );
            dims=:draw,
        ),
    )

    ndiv = count(sample_stats[:diverging])
    ndiv_perc = round(100 * ndiv//length(sample_stats[:diverging]); digits=2)
    ndiv > 0 &&
        @warn "$ndiv ($ndiv_perc%) divergent transitions encountered while sampling."

    # combine posterior and sample stats
    return InferenceObjects.InferenceData(; posterior, sample_stats)
end

function _dynamichmc_stats(result)
    draw_stats = map(result.tree_statistics) do stat
        termination = stat.termination
        return (
            energy=stat.π,
            tree_depth=stat.depth,
            acceptance_rate=stat.acceptance_rate,
            n_steps=stat.steps,
            diverging=termination.left == termination.right,
            turning=termination.left < termination.right,
        )
    end
    chain_stats = (; step_size=result.ϵ, inv_metric=result.κ.M⁻¹)
    return draw_stats, chain_stats
end

# utilities for separately running warmup and sampling for dynamichmc
# Note: technically these are not part of the API, but they are stable, see
# https://github.com/tpapp/DynamicHMC.jl/issues/177

function _extract_initialization(state)
    (; Q, κ, ϵ) = state
    return (; q=Q.q, κ, ϵ)
end

"""
    dhmc_warmup(rng::AbstractRNG, ℓ; kwargs...) -> NamedTuple

Run just the warmup stage of `DynamicHMC.mcmc_with_warmup`, returning a new `initialization`.

`kwargs` may contain any keyword argument accepted by `DynamicHMC.mcmc_with_warmup`, in
particular `warmup_stages` and `initialization`.
"""
function dhmc_warmup(
    rng::Random.AbstractRNG,
    ℓ;
    initialization=(),
    warmup_stages=DynamicHMC.default_warmup_stages(),
    kwargs...,
)
    initialization_final = foldl(warmup_stages; init=initialization) do init, stage
        result = DynamicHMC.mcmc_keep_warmup(
            rng, ℓ, 0; warmup_stages=(stage,), initialization=init, kwargs...
        )
        return _extract_initialization(result.final_warmup_state)
    end
    return initialization_final
end

"""
    dhmc_sample(rng::AbstractRNG, ℓ, ndraws; kwargs...) -> NamedTuple

Run the post-warmup stage of `DynamicHMC.mcmc_with_warmup`, returning `ndraws` samples.

`kwargs` may contain any keyword argument accepted by `DynamicHMC.mcmc_with_warmup`. See
that docstring for a description of the output.
"""
function dhmc_sample(rng::Random.AbstractRNG, ℓ, ndraws; initialization, kwargs...)
    return DynamicHMC.mcmc_with_warmup(
        rng, ℓ, ndraws; warmup_stages=(), initialization, kwargs...
    )
end

function _dhmc_warmup_until_succeeds(rng, prob, ntries; initialization=(), kwargs...)
    i = 1
    while i ≤ ntries
        try
            return dhmc_warmup(rng, prob; initialization, kwargs...), i
        catch e
            e isa DynamicHMC.DynamicHMCError || rethrow()
            @warn e
            @warn "DynamicHMC failed to sample, trying again ($i/$ntries)"
            i += 1
            (isempty(initialization) || !haskey(initialization, :q)) && continue
            q = DynamicHMC.random_position(rng, length(initialization.q))
            initialization = merge(initialization, (; q))
        end
    end
    @error "DynamicHMC failed to sample after $ntries tries"
end
