### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a0691b7a-a3b6-11ed-1bb2-0fcb8b9730f2
begin
    cd(@__DIR__)

    using Pkg
    Pkg.activate("..")
    using Revise
    using JLD2,
        LogDensityProblems,
        Logging,
        Pathfinder,
        PathfinderBenchmarks,
        PosteriorDB,
        ProgressLogging,
        Random,
        StanLogDensityProblems,
        TerminalLoggers,
        Transducers

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
    nchains = 4
    path = posterior_name
    isdir(path) || mkpath(path)
    post = PosteriorDB.posterior(pdb, posterior_name)
    prob = StanProblem(
        post, path; force=true, nan_on_error=true, make_args=["STAN_THREADS=true"]
    )
    dim = LogDensityProblems.dimension(prob)
    rng = Random.seed!(seed)
    run_seeds = rand(rng, UInt16, nruns)
    run_inits = rand(rng, dim, nchains, nruns) .* 4 .- 2
    @progress name = posterior_name for (benchmark_name, pathfinder_config) in
                                        all_pathfinder_configurations(dim)
        benchmark_path = joinpath(path, "diagnostic_$benchmark_name")
        benchmark_file = joinpath(benchmark_path, "results.jld2")
        isfile(benchmark_file) && continue
        isdir(benchmark_path) || mkpath(benchmark_path)
        run_files = String[]
        @progress name = benchmark_name for i in 1:nruns
            run_file = joinpath(benchmark_path, "run.$i.jld2")
            push!(run_files, run_file)
            isfile(run_file) && continue
            init = [run_inits[:, j, i] for j in axes(run_inits, 2)]
            Random.seed!(rng, run_seeds[i])
            result = multipathfinder(prob, 100; rng, init, pathfinder_config.options...)
            jldsave(run_file; result)
        end
        @info "combining runs"
        results = map(f -> jldopen(f)["result"], run_files)
        jldsave(benchmark_file; results)
    end
end

# ╔═╡ Cell order:
# ╠═a0691b7a-a3b6-11ed-1bb2-0fcb8b9730f2
# ╠═3cfbbc40-6189-4da9-8cea-5ed477b4f9d8
# ╠═372f1054-7b2b-431e-9903-cc130c0530cf
# ╠═87f31447-bf3c-4f7f-9ef1-1b2851ab00f0
