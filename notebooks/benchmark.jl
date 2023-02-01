### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# ╔═╡ 3e5b016a-8c82-11ed-3cde-d313502f9be0
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using DynamicHMC, LinearAlgebra, LogDensityProblems, Optim, Pathfinder, PathfinderBenchmarks, PosteriorDB, Random, Transducers
end

# ╔═╡ 6ee67094-215b-491e-bb8a-bcd3ef12b9c6
pdb = database()

# ╔═╡ 7746714a-d1f6-46d3-8383-d2fdfa2389ce
posterior_names = [
    "arma-arma11", "diamonds-diamonds", "eight_schools-eight_schools_centered"
]

# ╔═╡ e086ece5-9175-42be-93c0-939c3772088c
post = posterior(pdb, posterior_names[2])

# ╔═╡ b54ac5c9-6e2b-4f38-aa15-767f22251494
prob = StanProblem(post)

# ╔═╡ 1f9003ab-3a03-48af-90b5-fbbf76ffc920
idata = let
	warmup_stages = default_warmup_stages(; M=Symmetric)
	sample_dynamichmc(prob, 1_000, 4; warmup_stages)
end

# ╔═╡ c8ddbce9-01d2-4b19-a836-f8b7dbe83fac
pathfinder(prob)

# ╔═╡ Cell order:
# ╠═3e5b016a-8c82-11ed-3cde-d313502f9be0
# ╠═6ee67094-215b-491e-bb8a-bcd3ef12b9c6
# ╠═7746714a-d1f6-46d3-8383-d2fdfa2389ce
# ╠═e086ece5-9175-42be-93c0-939c3772088c
# ╠═b54ac5c9-6e2b-4f38-aa15-767f22251494
# ╠═1f9003ab-3a03-48af-90b5-fbbf76ffc920
# ╠═c8ddbce9-01d2-4b19-a836-f8b7dbe83fac
