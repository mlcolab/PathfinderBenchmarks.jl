### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 3e5b016a-8c82-11ed-3cde-d313502f9be0
begin
    cd(@__DIR__)

    using Pkg
    Pkg.activate("..")
    using Revise
    using BridgeStan,
        DynamicHMC,
        JSON3,
        LogDensityProblems,
        Logging,
        PathfinderBenchmarks,
        PosteriorDB,
        ProgressLogging,
        Random,
        StanLogDensityProblems,
        Statistics,
        TerminalLoggers

    Logging.global_logger(TerminalLogger(; right_justify=120))
end

# ╔═╡ aab0b1fd-af1b-4af8-af27-a15c03a45353
begin
    nruns = 10
    nchains = 4
    initial_ndraws = 1_000
    max_ndraws = 10_000
    target_ess_per_chain = 1_000
    initial_delta = 0.8
    max_delta = 0.99
    out_file = "hmc_config.json"
end;

# ╔═╡ 6ee67094-215b-491e-bb8a-bcd3ef12b9c6
pdb = PosteriorDB.database()

# ╔═╡ 7746714a-d1f6-46d3-8383-d2fdfa2389ce
posterior_seeds = [
    "arma-arma11" => 50872,
    "eight_schools-eight_schools_centered" => 33701,
    "diamonds-diamonds" => 18730,
    "mcycle_gp-accel_gp" => 39292,
];

# ╔═╡ 747bb969-a960-4b6e-bace-2062e4d0b2a7
target_ndraws = let
    @progress for (name, seed) in posterior_seeds
        isdir(name) || mkdir(name)
        config_file = joinpath(name, out_file)
        isfile(config_file) && continue
        rng = Random.seed!(seed)
        path = name
        post = PosteriorDB.posterior(pdb, path)
        prob = StanProblem(
            post, name; nan_on_error=true, force=true, make_args=["STAN_THREADS=true"]
        )
        ess_values = Vector{Float64}[]
        δ = initial_delta
        ndraws = initial_ndraws
        nruns_finished = 0
        ProgressLogging.progress() do id
            while nruns_finished < nruns
                Base.@logmsg ProgressLogging.ProgressLevel "running" progress =
                    (nruns_finished + 1) / nruns _id = id
                warmup_stages = default_warmup_stages(;
                    stepsize_adaptation=DualAveraging(; δ)
                )
                idata = sample_dynamichmc(prob, ndraws, nchains; rng, warmup_stages)
                y = idata.posterior.x
                x = mapslices(idata.posterior.x; dims=3) do xi
                    param_constrain(prob.model, vec(xi))
                end
                S, R = ess_rhat(x)
                frac_div = mean(idata.sample_stats.diverging)
                restart = false
                if (frac_div > 0.01) && (δ < max_delta)
                    δ = min(max_delta, δ + (1 - δ) / 2)
                    @warn "increasing δ to $δ (frac_div: $frac_div)"
                    restart = true
                elseif (any(<(100 * nchains), S) || any(>(1.01), R)) &&
                    (ndraws < max_ndraws)
                    ndraws = min(ndraws * 2, max_ndraws)
                    @warn "increasing ndraws to $ndraws (ess=$S, rhat=$R)"
                    restart = true
                end
                if restart
                    nruns_finished = 0
                    deleteat!(ess_values, eachindex(ess_values))
                else
                    push!(ess_values, S)
                    nruns_finished += 1
                end
            end
        end
        mean_ess_vals = mean(ess_values)
        ndraws_new = @. clamp(
            div(mean_ess_vals * target_ess_per_chain, ndraws),
            initial_ndraws,
            max_ndraws,
        )
        config = Dict(
            "nchains" => nchains,
            "ndraws" => Int(round(maximum(ndraws_new), RoundUp; digits=-2)),
            "delta" => δ,
        )
        @info "chosen config for $name: $config"
        open(config_file, "w") do io
            JSON3.write(io, config)
        end
    end
end

# ╔═╡ Cell order:
# ╠═3e5b016a-8c82-11ed-3cde-d313502f9be0
# ╠═aab0b1fd-af1b-4af8-af27-a15c03a45353
# ╠═6ee67094-215b-491e-bb8a-bcd3ef12b9c6
# ╠═7746714a-d1f6-46d3-8383-d2fdfa2389ce
# ╠═747bb969-a960-4b6e-bace-2062e4d0b2a7
