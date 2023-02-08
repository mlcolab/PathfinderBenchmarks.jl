# defining Pathfinder warmup stages
# Note: `warmup` and `report` are not API functions, so this code may break on future versions of DynamicHMC

function _pathfinder_point_and_metric(rng, ℓ, pathfinder_config, init)
    result = Pathfinder.pathfinder(ℓ; rng, init, ndraws=1, pathfinder_config.options...)
    return result.draws[:, 1], result.fit_distribution.Σ
end

"""
    PathfinderPointInitialization(cfg::PathfinderConfig)

Use `pathfinder` to draw an initial point for HMC.

This object can be included in the sequence of `warmup_stages` passed to
`DynamicHMC.mcmc_keep_warmup`.

If an initial point was provided to DynamicHMC, that will be used as the initial point for
constructing Pathfinder's trajectory.
"""
struct PathfinderPointInitialization{C<:PathfinderConfig}
    cfg::C
end

function DynamicHMC.warmup(
    sampling_logdensity, pf_init::PathfinderPointInitialization, warmup_state
)
    (; rng, ℓ, reporter) = sampling_logdensity
    (; q, κ, ϵ) = extract_initialization(warmup_state)
    q′, _ = _pathfinder_point_and_metric(rng, ℓ, pf_init.cfg, q)
    DynamicHMC.report(reporter, "Pathfinder drew initial point"; q=q′)
    return nothing, DynamicHMC.WarmupState(DynamicHMC.evaluate_ℓ(ℓ, q′), κ, ϵ)
end

"""
    PathfinderPointMetricInitialization(cfg::PathfinderConfig)

Use `pathfinder` to draw an initial point and metric for HMC.

This object can be included in the sequence of `warmup_stages` passed to
`DynamicHMC.mcmc_keep_warmup`.

If an initial point was provided to DynamicHMC, that will be used as the initial point for
constructing Pathfinder's trajectory. Any initial metric provided will be ignored.
"""
struct PathfinderPointMetricInitialization{C<:PathfinderConfig}
    cfg::C
end

function DynamicHMC.warmup(
    sampling_logdensity, pf_init::PathfinderPointMetricInitialization, warmup_state
)
    (; rng, ℓ, reporter) = sampling_logdensity
    (; q, ϵ) = extract_initialization(warmup_state)
    q′, M⁻¹ = _pathfinder_point_and_metric(rng, ℓ, pf_init.cfg, q)
    DynamicHMC.report(reporter, "Pathfinder drew initial point"; q=q′)
    DynamicHMC.report(reporter, "Pathfinder estimated inverse metric"; M⁻¹)
    κ = DynamicHMC.GaussianKineticEnergy(M⁻¹)
    return nothing, DynamicHMC.WarmupState(DynamicHMC.evaluate_ℓ(ℓ, q′), κ, ϵ)
end
