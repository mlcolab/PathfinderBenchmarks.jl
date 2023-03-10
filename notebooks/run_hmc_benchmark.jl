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
begin
    posterior_configs = [
        "arma-arma11" => (nruns=100, seed=5461),
        "eight_schools-eight_schools_centered" => (nruns=100, seed=2056),
        "diamonds-diamonds" => (nruns=20, seed=6842),
    ]
end

# ╔═╡ 87f31447-bf3c-4f7f-9ef1-1b2851ab00f0
@progress name = "posterior" for (posterior_name, config) in posterior_configs
    (; nruns, seed) = config
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
    @progress name = posterior_name for (benchmark_name, warmup_stages) in
                                        all_warmup_stages(dim, δ)
        benchmark_path = joinpath(path, "results_$benchmark_name")
        benchmark_file = joinpath(benchmark_path, "results.nc")
        isfile(benchmark_file) && continue
        isdir(benchmark_path) || mkpath(benchmark_path)
        run_files = String[]
        @progress name = benchmark_name for i in 1:nruns
            run_file = joinpath(benchmark_path, "run.$i.nc")
            push!(run_files, run_file)
            isfile(run_file) && continue
            initializations = [(; q=run_inits[:, j, i]) for j in axes(run_inits, 2)]
            Random.seed!(rng, run_seeds[i])
            idata = sample_dynamichmc(
                prob, ndraws, nchains; rng, initializations, warmup_stages
            )
            to_netcdf(idata, run_file)
        end
        @info "combining runs"
        idata_merged = cat(map(from_netcdf, run_files)...; dims=:run)
        to_netcdf(idata_merged, benchmark_file)
    end
end

# ╔═╡ Cell order:
# ╠═a0691b7a-a3b6-11ed-1bb2-0fcb8b9730f2
# ╠═3cfbbc40-6189-4da9-8cea-5ed477b4f9d8
# ╠═372f1054-7b2b-431e-9903-cc130c0530cf
# ╠═87f31447-bf3c-4f7f-9ef1-1b2851ab00f0
