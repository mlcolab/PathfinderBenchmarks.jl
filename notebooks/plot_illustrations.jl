### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ d75efdac-bdf4-11ed-1280-fdaceb1a5bd5
begin
    cd(@__DIR__)

    using Pkg
    Pkg.activate("..")
    using Revise
    using AlgebraOfGraphics,
		BridgeStan,
        CairoMakie,
        DataFrames,
        DimensionalData,
		InferenceObjects,
		KernelDensity,
        LinearAlgebra,
		LogDensityProblems,
		LogDensityProblemsAD,
		Optim,
		Pathfinder,
        PathfinderBenchmarks,
		PosteriorDB,
		Random,
		StanLogDensityProblems,
        Statistics,
		Transducers
    using AlgebraOfGraphics: density
	using Optim.LineSearches
end

# ╔═╡ 87b7d1b5-9714-48c9-a7e8-19c4973b14fd
let
	using Distributions
	# Σ = [1.0 -0.5; -0.5 1.0]
	# d = MvNormal(Σ)
	# x = rand(d, 1000)
	# fig = covellipseplot([0.0, 0.0], [1.0 -0.5; -0.5 1.0]; axis=(; aspect=DataAspect()), color=(:blue, 0.5), strokewidth=2, strokecolor=:black, n_std=2.45)
	# scatter!(x[1, :], x[2, :])
	# fig
	quantile(Chi(2), 0.95)
end

# ╔═╡ 0b10ea5e-6b28-43b2-b011-41b4ff3709f9
begin
    CairoMakie.activate!(; type="svg")
    set_theme!(theme_minimal())
    update_theme!(; fontsize=18, font="CMU", linewidth=2)
end;

# ╔═╡ 36f53f2b-9221-40b2-bbe8-aa54beaef3d2
function _covellipse_args(
    (μ, Σ)::Tuple{AbstractVector{<:Real},AbstractMatrix{<:Real}};
    n_std::Real,
)
    size(μ) == (2,) && size(Σ) == (2, 2) ||
        error("covellipse requires mean of length 2 and covariance of size 2×2.")
    λ, U = eigen(Σ)
    μ, n_std * U * diagm(.√λ)
end

# ╔═╡ ae675f63-367f-41c6-8bd7-4c1d44219242
begin
	@recipe(CovEllipsePlot, mean, cov) do scene
	    return Makie.Attributes(;
	        Makie.default_theme(scene, Makie.Poly)...,
			prob=0.95,
			n_ellipse_vertices=100,
	    )
	end

	function Makie.plot!(p::CovEllipsePlot)
	    attrs = p.attributes
		(; prob, n_ellipse_vertices) = attrs
	    μ, Σ = p[1], p[2]
		vertices = lift(μ, Σ, prob, n_ellipse_vertices) do μ, Σ, prob, n_ellipse_vertices
			λ, U = eigen(Σ)
			S = quantile(Chi(2), prob) .* U .* sqrt.(λ')
			θ = range(0, 2π; length = n_ellipse_vertices)
		    x = S * vcat(cos.(θ'), sin.(θ')) .+ μ
			map(Point2f, eachslice(x; dims=2))
		end
		ks = filter(∉((:prob, :n_ellipse_vertices)), keys(attrs))
	    return Makie.poly!(p, vertices; (k => attrs[k] for k in ks)...)
	end
end;

# ╔═╡ 8bbfb4f0-832d-4d64-96c7-d1116f90e5f2
begin
	struct BananaProblem end
	function LogDensityProblems.capabilities(::Type{<:BananaProblem})
	    return LogDensityProblems.LogDensityOrder{0}()
	end
	LogDensityProblems.dimension(ℓ::BananaProblem) = 2
	function LogDensityProblems.logdensity(ℓ::BananaProblem, x)
	    return -(x[1]^2 + 5(x[2] - x[1]^2)^2) / 2
	end
end;

# ╔═╡ 5bf1692e-1d8c-4da9-96d0-d9e7709b9ac2
prob_banana = ADgradient(Val(:ForwardDiff), BananaProblem())

# ╔═╡ f32a4d74-0e29-44a4-8608-f5bfd004aced
logp_banana(x) = LogDensityProblems.logdensity(prob_banana, x)

# ╔═╡ 0968b043-9a47-4770-bec6-4debae9309d7
pdb = PosteriorDB.database()

# ╔═╡ acc1a5ee-288e-4a64-9ffc-c260b0563a1c
eight_schools = let
	name = path = "eight_schools-eight_schools_centered"
	post = PosteriorDB.posterior(pdb, name)
	StanProblem(post, path; force=true, nan_on_error=true, make_args=["STAN_THREADS=true"])
end

# ╔═╡ 596d1b73-f187-487e-96e4-2aa61769892e
ref_draws = let
	draws = DimArray(PosteriorDB.load(PosteriorDB.reference_posterior(PosteriorDB.posterior(pdb, "eight_schools-eight_schools_centered"))), Dim{:chain})
	draws_arr = stack(map(draws) do chain
		DimArray(stack(collect(values(chain))), (:draw, Dim{:param}(BridgeStan.param_names(eight_schools.model))))
	end)
	draws_arr_unc = set(mapslices(draws_arr; dims=:param) do x
		BridgeStan.param_unconstrain(eight_schools.model, Vector(x))
	end, :param => Dim{:param_unc}(BridgeStan.param_unc_names(eight_schools.model)))
end

# ╔═╡ bcce9790-d4ce-472e-bf64-397168e0ecc3
alphaguess = InitialHagerZhang()

# ╔═╡ 2e6c7fc6-68bb-4fc2-a1b4-c305c1fa5253
linesearch = HagerZhang()

# ╔═╡ fa420ce7-2ff3-45f0-8025-9903fc3debd2
optimizer = LBFGS(; m=Pathfinder.DEFAULT_HISTORY_LENGTH, alphaguess, linesearch)

# ╔═╡ 9cb91345-ebce-495e-8a06-23c6466e51a7
result_banana = let
	rng = Random.seed!(8229)
	multipathfinder(prob_banana, 1_000; rng, ndraws_per_run=1000, nruns=1, init_scale=5, optimizer)
end

# ╔═╡ b3f03f01-47f1-4ba9-8b4b-53199a9264a0
let
	xrange = -3.5:0.01:3.5
	yrange = -3:0.01:7
	fig = Figure(; resolution=(600, 400))
	ax = Axis(fig[1, 1]; xlabel="x", ylabel="y")
	contourf!(ax, xrange, yrange, exp ∘ logp_banana ∘ Base.vect; levels = 0.05:0.05:1, mode=:relative, colormap=(:plasma, 0.25))
	result = result_banana.pathfinder_results[1]
	niter = length(result.fit_distributions)
	cov1 = nothing
	for (i, dist) in enumerate([result.fit_distributions[1], result.fit_distributions[2:3:(end-1)]..., result.fit_distributions[end]])
		(; μ, Σ) = dist
		cov1 = covellipseplot!(ax, μ, Σ, prob=0.95, color=(:black, 0.0), strokewidth=2, strokecolor=(:green, i / niter))
	end
	(; μ, Σ) = result.fit_distribution
	cov2 = covellipseplot!(ax, μ, Σ, prob=0.95, color=(:black, 0.0), strokewidth=2, strokecolor=:orange)
	tracex = first.(result.optim_trace.points)
	tracey = last.(result.optim_trace.points)
	scat = scatter!(ax, result.draws[1, :], result.draws[2, :], color=(:gray20, 0.2), markersize=5)
	scat2 = scatter!(ax, result_banana.draws[1, :], result_banana.draws[2, :], color=(:black, 0.8), markersize=5)
	lintrace = scatterlines!(ax, tracex, tracey, color=:blue, markersize=7)
	xlims!(ax, -2.5, 2.5)
	ylims!(ax, -2.1, 6.5)
	hidedecorations!(ax)
	hidespines!(ax)
	Legend(fig[1, 1], [lintrace, cov1, cov2, scat, scat2], ["L-BFGS path", "fit along path", "final fit", "draws", "resampled draws"]; labelsize=14, orientation=:horizontal, tellheight=false, tellwidth=false, valign=:top)
	Makie.save("banana_illustration.pdf", fig)
	fig
end

# ╔═╡ ea67d8bc-c272-4606-9a64-75cd6bbd8b9b
result_banana.pathfinder_results[1].optim_trace.points

# ╔═╡ 2bb6ddd9-2688-48d3-b7a8-e6067cf2cd35
result_eight = let
	rng = Random.seed!(2002)
	nruns = 8
	init_scale = 10
	executor = ThreadedEx()
	multipathfinder(eight_schools, 1_000; rng, executor, nruns, init_scale, optimizer)
end

# ╔═╡ a581f238-820b-4729-9788-73a4d1086625
draws = DimArray(result_eight.draws, (Dimensions.dims(ref_draws, :param_unc), :draw))

# ╔═╡ f59e744b-9d80-4591-8fe1-f62d94bbe99f
let
	θ1 = draws[param_unc=At("theta.1")]
	τ = draws[param_unc=At("tau")]
	θ1ref = ref_draws[param_unc=At("theta.1")]
	τref = ref_draws[param_unc=At("tau")]
	fig = Figure(; resolution = (800, 500))
	axis = Axis(fig[1, 1]; title="eight_schools_centered", xlabel=L"\theta_1", ylabel=L"\log\tau")
	cont = contourf!(axis, kde(hcat(vec(θ1ref), vec(τref))); levels = 0.1:0.05:1, mode=:relative, colormap=(:plasma, 0.25))
	scat = scatter!(axis, vec(θ1), vec(τ); color=(:black, 0.4), markersize=6)
	xlims!(axis, -7, 22)
	ylims!(axis, -2.5, 2.7)
	Legend(fig[1, 1], [cont, scat], ["reference draws", "Pathfinder draws"]; tellwidth=false, tellheight=false, valign=:bottom, halign=:right, padding=(0, 30, 30, 0))
	Makie.save("eight_schools_scatter.pdf", fig)
	fig
end

# ╔═╡ Cell order:
# ╠═d75efdac-bdf4-11ed-1280-fdaceb1a5bd5
# ╠═0b10ea5e-6b28-43b2-b011-41b4ff3709f9
# ╠═36f53f2b-9221-40b2-bbe8-aa54beaef3d2
# ╠═ae675f63-367f-41c6-8bd7-4c1d44219242
# ╠═87b7d1b5-9714-48c9-a7e8-19c4973b14fd
# ╠═8bbfb4f0-832d-4d64-96c7-d1116f90e5f2
# ╠═5bf1692e-1d8c-4da9-96d0-d9e7709b9ac2
# ╠═f32a4d74-0e29-44a4-8608-f5bfd004aced
# ╠═9cb91345-ebce-495e-8a06-23c6466e51a7
# ╠═b3f03f01-47f1-4ba9-8b4b-53199a9264a0
# ╠═ea67d8bc-c272-4606-9a64-75cd6bbd8b9b
# ╠═0968b043-9a47-4770-bec6-4debae9309d7
# ╠═acc1a5ee-288e-4a64-9ffc-c260b0563a1c
# ╠═596d1b73-f187-487e-96e4-2aa61769892e
# ╠═bcce9790-d4ce-472e-bf64-397168e0ecc3
# ╠═2e6c7fc6-68bb-4fc2-a1b4-c305c1fa5253
# ╠═fa420ce7-2ff3-45f0-8025-9903fc3debd2
# ╠═2bb6ddd9-2688-48d3-b7a8-e6067cf2cd35
# ╠═a581f238-820b-4729-9788-73a4d1086625
# ╠═f59e744b-9d80-4591-8fe1-f62d94bbe99f
