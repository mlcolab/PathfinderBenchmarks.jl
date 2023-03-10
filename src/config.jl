"""
    PathfinderConfig(; options...)

A configuration object for `Pathfinder.pathfinder`.

Wherever this object is used, `options` are passed as keyword arguments to `pathfinder`.
"""
struct PathfinderConfig{O<:NamedTuple}
    options::O
end
PathfinderConfig(; options...) = PathfinderConfig(NamedTuple(options))

# all pathfinder configurations to be benchmarked.
# TODO: programatically generate this list from lists of combinations.
function all_pathfinder_configurations(n::Int)
    return [
        "default" => PathfinderConfig(),
        "initstaticscaled" => PathfinderConfig(;
            optimizer=Optim.LBFGS(;
                alphaguess=LineSearches.InitialStatic(; scaled=true),
                linesearch=LineSearches.MoreThuente(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
            ),
        ),
        "hagerzhangls" => PathfinderConfig(;
            optimizer=Optim.LBFGS(;
                linesearch=LineSearches.HagerZhang(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
            ),
        ),
        "hagerzhangls_gilbertinit" => PathfinderConfig(;
            optimizer=LBFGS(;
                linesearch=LineSearches.HagerZhang(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
                init_invH0=init_invH0_gilbert!,
            ),
        ),
        "inithagerzhangls_hagerzhangls" => PathfinderConfig(;
            optimizer=Optim.LBFGS(;
                alphaguess=LineSearches.InitialHagerZhang(),
                linesearch=LineSearches.HagerZhang(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
            ),
        ),
        "initstaticscaled_hagerzhangls" => PathfinderConfig(;
            optimizer=Optim.LBFGS(;
                alphaguess=LineSearches.InitialStatic(; scaled=true),
                linesearch=LineSearches.HagerZhang(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
            ),
        ),
        "inithagerzhangls_gilbertinit" => PathfinderConfig(;
            optimizer=LBFGS(;
                alphaguess=LineSearches.InitialHagerZhang(),
                linesearch=LineSearches.MoreThuente(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
                init_invH0=init_invH0_gilbert!,
            ),
        ),
        "gilbertinit" => PathfinderConfig(;
            optimizer=LBFGS(;
                linesearch=LineSearches.MoreThuente(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
                init_invH0=init_invH0_gilbert!,
            ),
        ),
        "initstaticscaled_gilbertinit" => PathfinderConfig(;
            optimizer=LBFGS(;
                alphaguess=LineSearches.InitialStatic(; scaled=true),
                linesearch=LineSearches.MoreThuente(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
                init_invH0=init_invH0_gilbert!,
            ),
        ),
        "nloptlbfgs" => PathfinderConfig(; optimizer=NLopt.Opt(:LD_LBFGS, n)),
        "backtrackingls" => PathfinderConfig(;
            optimizer=Optim.LBFGS(;
                linesearch=LineSearches.BackTracking(),
                m=Pathfinder.DEFAULT_HISTORY_LENGTH,
            ),
        ),
    ]
end

# all warmup stages to be benchmarked.
function all_warmup_stages(n::Int, δ::Real)
    stepsize_adaptation = DynamicHMC.DualAveraging(; δ)
    dws_diag = DynamicHMC.default_warmup_stages(; stepsize_adaptation)
    dws_dense = DynamicHMC.default_warmup_stages(; stepsize_adaptation, M=Symmetric)
    dws_none = DynamicHMC.default_warmup_stages(;
        stepsize_adaptation, doubling_stages=0, middle_steps=0
    )[1:(end - 1)]  # only keep one step size adaptation stage
    pf_default_cfg = PathfinderConfig()
    configs = Pair[
        "default_diag" => dws_diag,
        "default_dense" => dws_dense,
        "pathfinder_point_init" => (
            PathfinderPointInitialization(pf_default_cfg), dws_diag...
        ),
        "pathfinder_metric_diag_init" => (
            PathfinderPointMetricInitialization(pf_default_cfg), dws_diag...
        ),
        "pathfinder_metric_dense_init" => (
            PathfinderPointMetricInitialization(pf_default_cfg), dws_dense...
        ),
    ]
    prefix = "pathfinder_metric"
    for (name, pf_cfg) in all_pathfinder_configurations(n)
        full_name = name == "default" ? prefix : "$(prefix)_$name"
        push!(
            configs, full_name => (PathfinderPointMetricInitialization(pf_cfg), dws_none...)
        )
    end
    return configs
end
