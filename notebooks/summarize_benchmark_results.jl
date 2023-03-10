### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a791fbea-a7e4-11ed-0c1e-0f45961d3d26
begin
    using Pkg
    Pkg.activate("..")
    using Revise
    using BridgeStan,
        DataFrames,
        DimensionalData,
        JLD2,
        InferenceObjects,
        InferenceObjectsNetCDF,
        LinearAlgebra,
        MCMCDiagnosticTools,
        PathfinderBenchmarks,
        PosteriorDB,
	    StanLogDensityProblems,
        Statistics
end

# ╔═╡ f7084e2e-196e-41c9-b791-28fff8c4b0ae
function load_reference_draws(post, prob)
	ref = PosteriorDB.reference_posterior(post)
	dicts = PosteriorDB.load(ref)
	ds = map(dicts) do d
		inds = [replace(k, '[' => '.', ']' => "") for k in keys(d)]
		vals = reduce(hcat, values(d))
		vals = reshape(vals, :, 1, size(vals, 2))
		return convert_to_dataset(vals; coords=(; param=inds), dims=(; x=[:param]))
	end
	x = cat(ds...; dims=Dim{:chain}).x
	x_unc = mapslices(x -> param_unconstrain(prob.model, x), x; dims=:param)
	x_unc = set(x_unc; param=Dim{:param_unc}(param_unc_names(prob.model)))
	return Dataset((; x, x_unc))
end;

# ╔═╡ 97869614-fb67-4373-acf6-6574d6558a92
function format_chains_as(ref, target)
	nsample = size(ref, :draw) * size(ref, :chain)
	ndraws = size(target, :draw)
	nchains = div(nsample, ndraws)
	nsample_new = nchains * ndraws
	sz_new = (ndraws, nchains, :)
	ref_new = reshape(view(reshape(parent(ref), nsample, :), 1:nsample_new, :), sz_new)
	return DimArray(ref_new, (:draw, :chain, :param))
end;

# ╔═╡ 4262101c-26b8-4f4b-9ce8-848a1138f77b
function load_hmc_benchmark_results(model_path, benchmarks=nothing)
    results_file = joinpath(model_path, "all_results.nc")
    idata = if isfile(results_file)
        from_netcdf(results_file)
    else
        subdirs = filter(startswith("results_"), readdir(model_path))
        run_idatas = Pair{String,InferenceData}[]
        for subdir in subdirs
            name = subdir[9:end]
            path = joinpath(model_path, subdir, "results.nc")
            isfile(path) || continue
            push!(run_idatas, name => from_netcdf(path))
        end
        idata = cat(map(last, run_idatas)...; dims=Dim{:benchmark}(map(first, run_idatas)))
        to_netcdf(idata, results_file)
        idata
	end
	benchmarks === nothing && return idata
    return idata[benchmark=At(benchmarks)]
end

# ╔═╡ 031188ea-f075-47c8-9417-ce8f9ae3ce79
function compute_rhat_vs_reference(idata, var_name=:x)
	posterior = idata.posterior
	ref_posterior = idata.reference_posterior
	rhat_vals = stack(
		map(eachslice(posterior[var_name]; dims=(:run, :benchmark))) do xi
			xref = format_chains_as(ref_posterior[var_name], xi)
			rhat(cat(xi, xref; dims=Dim{:chain}))
		end,
	)
	return Dataset(NamedTuple{(var_name,)}((rhat_vals,)))
end;

# ╔═╡ 433e239b-9fb2-4fde-b6d8-c1e18cae7091
function compute_performance_metrics(converg_diag, sample_stats)
	num_evals_total = dropdims(
		sum(sample_stats.num_evals; dims=(:chain, :eval_type, :stage));
		dims=(:chain, :eval_type, :stage),
	)
	time_total = dropdims(sum(sample_stats.time; dims=(:chain, :stage)); dims=(:chain, :stage))
	denom = cat(num_evals_total, time_total; dims=Dim{:per}(["num_evals", "time"]))
	return broadcast_dims(/, converg_diag, denom)
end

# ╔═╡ 7b578f06-5d17-436d-b844-fc8cfb927a67
function compute_hmc_stats(idata)
    frac_diverging = dropdims(
        mean(idata.sample_stats.diverging; dims=(:draw, :chain, :run));
        dims=(:draw, :chain, :run),
    )
	convergence_diag = cat(
		compute_rhat_vs_reference(idata),
		ess_rhat(idata),
		ess(idata; kind=:tail);
		dims=Dim{:_metric}(["rhat_vs_ref", "ess_bulk", "rhat", "ess_tail"]),
	).x
	bfmi_vals = dropdims(
		mapslices(bfmi, idata.sample_stats.energy; dims=:draw);
		dims=:draw,
	)
	performance = compute_performance_metrics(convergence_diag, idata.sample_stats)
	condition_number_metric = rebuild(
	    dropdims(
	        mapslices(
	            pdcond,
	            idata.sample_stats.inv_metric;
	            dims=(:inv_metric_dim_1, :inv_metric_dim_2),
	        );
	        dims=(:inv_metric_dim_1, :inv_metric_dim_2),
	    );
	)
	return Dataset((; frac_diverging, convergence_diag, bfmi=bfmi_vals, performance, condition_number_metric))
end

# ╔═╡ 3358111c-8c4b-498f-a935-a2046135587f
function compute_condition_numbers(x; dims)
	cond_full = map(cond, eachslice(x; dims))
	cond_diag = map(cond ∘ Diagonal, eachslice(x; dims))
	cond_ratio = cond_full ./ cond_diag
	cond_vals = cat(
		cond_full, cond_diag, cond_ratio; dims=Dim{:cond}(["full", "diag", "ratio"]),
	)
	return cond_vals
end;

# ╔═╡ 2136a6b2-528b-4032-810c-db4d9895d2c1
function compute_reference_posterior_stats(ref, prob)
	model = prob.model
	sample_dims = Dimensions.dims(ref, (:draw, :chain))
	param_dims = Dimensions.dims(ref, :param_unc)
	param_dims2 = Dimensions.Dim{:param_unc2}(DimensionalData.lookup(param_dims))
	x = PermutedDimsArray(ref.x_unc, (sample_dims..., param_dims))
	x_reshape = reshape(parent(x), :, size(ref, param_dims))
	cov_unc = DimArray(cov(x_reshape), (param_dims, param_dims2))
	hessian = stack(map(eachslice(ref.x_unc; dims=(:draw, :chain))) do xi
		_, _, H = BridgeStan.log_density_hessian(prob.model, Vector(xi))
		return DimArray(H, (param_dims, param_dims2))
	end)
	# Note: cond(inv(X)) == cond(X)
	condition_inv_hessian = compute_condition_numbers(hessian; dims=(:draw, :chain))
	return Dataset((; cov_unc, hessian, condition_inv_hessian))
end;

# ╔═╡ 6243b239-0347-41f6-b9f2-41fd00c06bca
function load_pathfinder_benchmark_results(model_path)
	prefix = "diagnostic_"
	subdirs = filter(startswith(prefix), readdir(model_path))
	pathfinder_results = Pair[]
	for subdir in subdirs
		name = subdir[(length(prefix) + 1):end]
		path = joinpath(model_path, subdir, "results.jld2")
		isfile(path) || continue
		push!(pathfinder_results, name => jldopen(path)["results"])
	end
	return pathfinder_results
end;

# ╔═╡ 24fd01ce-6eff-4b16-88c5-116b9fbf029c
function pathfinder_results_to_dataset(pathfinder_results, prob)
	benchmarks = map(first, pathfinder_results)
	param_names = BridgeStan.param_names(prob.model)
	param_unc_names = BridgeStan.param_unc_names(prob.model)
	dims = (
		x_unc_resampled=(:param_unc, :draw),
		x_resampled=(:param, :draw),
		x_unc=(:param_unc, :draw_single),
		x=(:param, :draw_single),
		μ=(:param_unc,),
		Σ=(:param_unc, :param_unc2),
	)
	coords = (
		param_unc=param_unc_names,
		param_unc2=param_unc_names,
		param=param_names,
	)
	ds = cat(map(pathfinder_results) do (name, runs)
		data = cat(map(runs) do mpf
			single_path_ds = cat(map(mpf.pathfinder_results) do spf
				(; draws, fit_distribution, elbo_estimates, fit_iteration, num_bfgs_updates_rejected, num_tries) = spf
				x_unc = draws
				x = mapslices(xi -> BridgeStan.param_constrain(prob.model, xi), x_unc; dims=1)
				μ = fit_distribution.μ
				Σ = fit_distribution.Σ
				elbo = elbo_estimates[fit_iteration].value
				namedtuple_to_dataset(
					(; x_unc, x, μ, Σ, fit_iteration, num_bfgs_updates_rejected, num_tries, elbo);
					dims,
					coords,
					default_dims=(),
				)
			end...; dims=:pf_chain)
			x_unc_resampled = mpf.draws
			x_resampled = mapslices(xi -> BridgeStan.param_constrain(prob.model, xi), x_unc_resampled; dims=1)
			pareto_shape = mpf.psis_result.pareto_shape
			merge(
				single_path_ds,
				namedtuple_to_dataset((; x_unc_resampled, x_resampled, pareto_shape); dims, coords, default_dims=()),
			)
		end...; dims=:run)
	end...; dims=Dim{:benchmark}(benchmarks))
	param_dim = Dim{:param}(param_names)
	ds
end;

# ╔═╡ 79935e22-1d56-48ad-a786-a910d554f0fc
function compute_pathfinder_stats(pathfinder_results)
	Σ = pathfinder_results.Σ
	param_unc_dim = DimensionalData.dims(Σ, :param_unc)
	condition_inv_hessian = compute_condition_numbers(
		Σ; dims=otherdims(Σ, (:param_unc, :param_unc2)),
	)
	Dataset((; condition_inv_hessian))
end;

# ╔═╡ 2fe544c5-66f9-448b-8b53-aed209de8a52
pdb = PosteriorDB.database();

# ╔═╡ b45378f6-5a1c-4117-a2ee-0b8809283bec
posterior_names = [
	"arma-arma11",
	"eight_schools-eight_schools_centered",
	"diamonds-diamonds",
];

# ╔═╡ ca30d0a6-500e-48fc-94ce-27e3dc59a0ac
stan_probs = let
	Dict(map(posterior_names) do name
		path = name
	    post = PosteriorDB.posterior(pdb, name)
	    prob = StanProblem(
	        post, path; force=true, nan_on_error=true, make_args=["STAN_THREADS=true"]
	    )
		return name => prob
	end)
end

# ╔═╡ 3b3a88f8-89b9-4a8b-83aa-e594be733a87
function summarize_hmc_benchmark(name)
	post = PosteriorDB.posterior(pdb, name)
	prob = stan_probs[name]
	reference_posterior = load_reference_draws(post, prob)
	hmc_results = load_hmc_benchmark_results(name)
	idata = merge(hmc_results, InferenceData(; reference_posterior))
	hmc_benchmark_stats = compute_hmc_stats(idata)
	reference_posterior_stats = compute_reference_posterior_stats(
		reference_posterior, prob,
	)
	idata = merge(idata, InferenceData(; hmc_benchmark_stats, reference_posterior_stats))
	return idata
end;

# ╔═╡ cb04bda8-4e39-497c-8ceb-7287dd7ffaf1
function summarize_pathfinder_benchmark(name)
	path = name
	results = load_pathfinder_benchmark_results(path)
	pathfinder_results = pathfinder_results_to_dataset(results, stan_probs[name])
	pathfinder_benchmark_stats = compute_pathfinder_stats(pathfinder_results)
	return InferenceData(; pathfinder_results, pathfinder_benchmark_stats)
end;

# ╔═╡ e585a36a-c3eb-4f0c-8dda-8f944e52d66b
function summarize_benchmark(name)
	return merge(
		summarize_hmc_benchmark(name),
		summarize_pathfinder_benchmark(name),
	)
end;

# ╔═╡ caad32f9-999a-40d0-9d1c-8d41a09e4998
foreach(posterior_names) do name
	path = name
	idata = summarize_benchmark(name)
	to_netcdf(idata, joinpath(path, "benchmark_summary.nc"))
end;

# ╔═╡ Cell order:
# ╠═a791fbea-a7e4-11ed-0c1e-0f45961d3d26
# ╠═f7084e2e-196e-41c9-b791-28fff8c4b0ae
# ╠═97869614-fb67-4373-acf6-6574d6558a92
# ╠═4262101c-26b8-4f4b-9ce8-848a1138f77b
# ╠═031188ea-f075-47c8-9417-ce8f9ae3ce79
# ╠═433e239b-9fb2-4fde-b6d8-c1e18cae7091
# ╠═7b578f06-5d17-436d-b844-fc8cfb927a67
# ╠═3358111c-8c4b-498f-a935-a2046135587f
# ╠═2136a6b2-528b-4032-810c-db4d9895d2c1
# ╠═3b3a88f8-89b9-4a8b-83aa-e594be733a87
# ╠═6243b239-0347-41f6-b9f2-41fd00c06bca
# ╠═24fd01ce-6eff-4b16-88c5-116b9fbf029c
# ╠═79935e22-1d56-48ad-a786-a910d554f0fc
# ╠═cb04bda8-4e39-497c-8ceb-7287dd7ffaf1
# ╠═e585a36a-c3eb-4f0c-8dda-8f944e52d66b
# ╠═2fe544c5-66f9-448b-8b53-aed209de8a52
# ╠═b45378f6-5a1c-4117-a2ee-0b8809283bec
# ╠═ca30d0a6-500e-48fc-94ce-27e3dc59a0ac
# ╠═caad32f9-999a-40d0-9d1c-8d41a09e4998
