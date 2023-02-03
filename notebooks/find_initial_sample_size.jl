### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# ╔═╡ 3e5b016a-8c82-11ed-3cde-d313502f9be0
begin
    using Pkg
    Pkg.activate("..")
    using Revise
    using BridgeStan,
		DynamicHMC,
	    JSON3,
        LogDensityProblems,
        PathfinderBenchmarks,
        PosteriorDB,
	    ProgressLogging,
        Random,
	    StanLogDensityProblems,
	    Statistics
end

# ╔═╡ aab0b1fd-af1b-4af8-af27-a15c03a45353
begin
	nruns = 10
	nchains = 4
	ndraws = 1_000
	target_ess_per_chain = 1_000
	out_file = "hmc_config.json"
end;

# ╔═╡ 6ee67094-215b-491e-bb8a-bcd3ef12b9c6
pdb = database()

# ╔═╡ 7746714a-d1f6-46d3-8383-d2fdfa2389ce
posterior_names = [
    "arma-arma11", "eight_schools-eight_schools_centered", "diamonds-diamonds",
]

# ╔═╡ 747bb969-a960-4b6e-bace-2062e4d0b2a7
target_ndraws = let
	rng = Random.seed!(87992)
	target_ndraws = Dict()
	@progress for name in posterior_names
		post = posterior(pdb, name)
		prob = StanProblem(post; nan_on_error=true, make_args=["STAN_THREADS=true"])
		ess_values = Vector{Float64}[]
		@progress for _ in 1:nruns
			idata = sample_dynamichmc(prob, ndraws, nchains; rng)
			y = idata.posterior.x
			x = mapslices(idata.posterior.x; dims=3) do xi
				param_constrain(prob.model, vec(xi))
			end
			S, _ = ess_rhat(x)
			push!(ess_values, S)
		end
		mean_ess_vals = mean(ess_values)
		ndraws_new = @.(max(ndraws, Int(cld(mean_ess_vals * target_ess_per_chain, ndraws))))
		target_ndraws[name] = Dict(
			"nchains" => nchains,
			"ndraws" => Int(round(maximum(ndraws_new), RoundUp; digits=-2)),
		)
	end
	target_ndraws
end

# ╔═╡ 8275284a-e96e-4ec8-bebf-20c0a8388e9c
for (name, config) in pairs(target_ndraws)
	isdir(name) || mkdir(name)
	open(joinpath(name, out_file), "w") do io
		JSON3.write(io, config)
	end
end

# ╔═╡ Cell order:
# ╠═3e5b016a-8c82-11ed-3cde-d313502f9be0
# ╠═aab0b1fd-af1b-4af8-af27-a15c03a45353
# ╠═6ee67094-215b-491e-bb8a-bcd3ef12b9c6
# ╠═7746714a-d1f6-46d3-8383-d2fdfa2389ce
# ╠═747bb969-a960-4b6e-bace-2062e4d0b2a7
# ╠═8275284a-e96e-4ec8-bebf-20c0a8388e9c
