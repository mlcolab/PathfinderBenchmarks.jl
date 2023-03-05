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
        Logging,
        Optim,
        OptimizationNLopt,
        Pathfinder,
        PathfinderBenchmarks,
        PosteriorDB,
        ProgressLogging,
        Random,
        StanLogDensityProblems,
        Statistics,
        TerminalLoggers,
        Transducers
    using Optim.LineSearches

    Logging.global_logger(TerminalLogger(; right_justify=120))
end

# ╔═╡ 3cfbbc40-6189-4da9-8cea-5ed477b4f9d8
pdb = PosteriorDB.database()

# ╔═╡ 372f1054-7b2b-431e-9903-cc130c0530cf
posterior_seeds = [
    "arma-arma11" => 5461,
    "eight_schools-eight_schools_centered" => 2056,
    "diamonds-diamonds" => 6842,
];

# ╔═╡ 87f31447-bf3c-4f7f-9ef1-1b2851ab00f0
@progress name = "posterior" for (posterior_name, seed) in posterior_seeds
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
    δ = hmc_config["delta"]

    rng = Random.seed!(seed)
    run_seeds = rand(rng, UInt16, nruns)
    run_inits = rand(rng, dim, nchains, nruns) .* 4 .- 2
    @progress name = posterior_name for (run_name, warmup_stages) in
                                        all_warmup_stages(dim, δ)
        run_path = joinpath(path, "results_$run_name")
        isdir(run_path) || mkpath(run_path)
        @progress name = run_name for i in 1:nruns
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
# ╠═3cfbbc40-6189-4da9-8cea-5ed477b4f9d8
# ╠═372f1054-7b2b-431e-9903-cc130c0530cf
# ╠═87f31447-bf3c-4f7f-9ef1-1b2851ab00f0
