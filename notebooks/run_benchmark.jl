### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a0691b7a-a3b6-11ed-1bb2-0fcb8b9730f2
begin
    using Pkg
    Pkg.activate("..")
    using Revise
    using BridgeStan,
        DynamicHMC,
        InferenceObjectsNetCDF,
        JSON3,
        LinearAlgebra,
        LogDensityProblems,
        Optim,
        Pathfinder,
        PathfinderBenchmarks,
        PosteriorDB,
        ProgressLogging,
        Random,
        StanLogDensityProblems,
        Statistics,
        Transducers
end

# ╔═╡ 99590696-7880-4835-aa3d-bdbb39da71d7
begin
    nruns = 20
    all_warmup_stages = [
        "default_diag" => default_warmup_stages(),
        "default_dense" => default_warmup_stages(; M=Symmetric),
        "pathfinder_point_init" =>
            (PathfinderPointInitialization(PathfinderConfig()), default_warmup_stages()...),
        "pathfinder_metric_diag_init" => (
            PathfinderPointMetricInitialization(PathfinderConfig()),
            default_warmup_stages()...,
        ),
        "pathfinder_metric_dense_init" => (
            PathfinderPointMetricInitialization(PathfinderConfig()),
            default_warmup_stages(; M=Symmetric)...,
        ),
        "pathfinder_metric" => (
            PathfinderPointMetricInitialization(PathfinderConfig()),
            default_warmup_stages(; doubling_stages=0, middle_steps=0)[1:(end - 1)]...,
        ),
    ]
end;

# ╔═╡ 3cfbbc40-6189-4da9-8cea-5ed477b4f9d8
pdb = PosteriorDB.database()

# ╔═╡ 372f1054-7b2b-431e-9903-cc130c0530cf
posterior_seeds = [
    "arma-arma11" => 5461,
    "eight_schools-eight_schools_centered" => 2056,
    "diamonds-diamonds" => 6842,
];

# ╔═╡ 87f31447-bf3c-4f7f-9ef1-1b2851ab00f0
@progress for (posterior_name, seed) in posterior_seeds
    startswith(posterior_name, "arma-arma11") || continue
    path = posterior_name
    isdir(path) || mkpath(path)
    post = PosteriorDB.posterior(pdb, posterior_name)
    prob = StanProblem(
        post, path; force=true, nan_on_error=true, make_args=["STAN_THREADS=true"]
    )
    dim = LogDensityProblems.dimension(prob)
    hmc_config = open(JSON3.read, joinpath(path, "hmc_config.json"))
    ndraws = hmc_config["ndraws"]
    nchains = hmc_config["nchains"]

    rng = Random.seed!(seed)
    run_seeds = rand(rng, UInt16, nruns)
    run_inits = rand(rng, dim, nchains, nruns) .* 4 .- 2
    @progress for (run_name, warmup_stages) in all_warmup_stages
        run_path = joinpath(path, "results_$run_name")
        isdir(run_path) || mkpath(run_path)
        @progress for i in 1:nruns
            run_file = joinpath(run_path, "run.$i.nc")
            isfile(run_file) && continue
            initializations = [(; q=run_inits[:, j, i]) for j in axes(run_inits, 2)]
            Random.seed!(rng, run_seeds[i])
            idata = sample_dynamichmc(
                prob, ndraws, nchains; rng, initializations, warmup_stages
            )
            to_netcdf(idata, run_file)
        end
    end
end

# ╔═╡ Cell order:
# ╠═a0691b7a-a3b6-11ed-1bb2-0fcb8b9730f2
# ╠═99590696-7880-4835-aa3d-bdbb39da71d7
# ╠═3cfbbc40-6189-4da9-8cea-5ed477b4f9d8
# ╠═372f1054-7b2b-431e-9903-cc130c0530cf
# ╠═87f31447-bf3c-4f7f-9ef1-1b2851ab00f0
