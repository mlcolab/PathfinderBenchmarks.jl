### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a791fbea-a7e4-11ed-0c1e-0f45961d3d26
begin
    using Pkg
    Pkg.activate("..")
    using Revise
    using AlgebraOfGraphics,
        CairoMakie,
        DataFrames,
        DimensionalData,
        InferenceObjects,
        InferenceObjectsNetCDF,
        LinearAlgebra,
        MCMCDiagnosticTools,
        PathfinderBenchmarks,
        PosteriorDB,
        Statistics
    using AlgebraOfGraphics: density
end

# ╔═╡ 6ab900d1-008c-4cde-b114-408c1adcea33
begin
    CairoMakie.activate!(; type="svg")
    set_theme!(theme_minimal())
    update_theme!(; fontsize=18, font="CMU", linewidth=2)
    axis_ecdf = (
        width=500,
        height=450,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(10),
        ylabel="probability",
    )
end;

# ╔═╡ f7084e2e-196e-41c9-b791-28fff8c4b0ae
function load_reference_draws(post)
    ref = PosteriorDB.reference_posterior(post)
    dicts = PosteriorDB.load(ref)
    ds = map(dicts) do d
        inds = [replace(k, '[' => '.', ']' => "") for k in keys(d)]
        vals = reduce(hcat, values(d))
        vals = reshape(vals, :, 1, size(vals, 2))
        return convert_to_dataset(vals; coords=(; param=inds), dims=(; x=[:param]))
    end
    return cat(ds...; dims=Dim{:chain})
    # xnew = DimArray(reshape(parent(xcat), :, 1, size(xcat, 3)), (:draw, :chain, :param))
    # return Dataset((; x=xnew))
end;

# ╔═╡ b45378f6-5a1c-4117-a2ee-0b8809283bec
posterior_name = "arma-arma11"

# ╔═╡ 144af9d2-a6de-4c8c-8feb-bbe61d6c28e8
ref_draws = let
    pdb = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, posterior_name)
    ref = load_reference_draws(post)
end

# ╔═╡ 2dab771d-2d8e-4603-8aa9-7a7f02e98b41
function format_chains_as(ref, target)
    nsample = size(ref, :draw) * size(ref, :chain)
    ndraws = size(target, :draw)
    nchains = div(nsample, ndraws)
    nsample_new = nchains * ndraws
    sz_new = (ndraws, nchains, :)
    ref_new = reshape(view(reshape(parent(ref), nsample, :), 1:nsample_new, :), sz_new)
    return DimArray(ref_new, (:draw, :chain, :param))
end

# ╔═╡ 88f26eb3-8ca5-4146-aa66-8ae9ecd440a9
idata = let
    model_path = posterior_name
    results_file = joinpath(model_path, "all_results.nc")
    idata = if false  # isfile(results_file)
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
    benchmarks = sort([
        "default_dense",
        "default_diag",
        "pathfinder_metric",
        "pathfinder_metric_backtrackingls",
        "pathfinder_metric_hagerzhangls",
        # "pathfinder_metric_inithagerzhangls_hagerzhangls",
        # "pathfinder_metric_initstaticscaled_hagerzhangls",
        # "pathfinder_metric_inithagerzhangls_gilbertinit",
        # "pathfinder_metric_hagerzhangls_gilbertinit",
        "pathfinder_metric_initstaticscaled",
        "pathfinder_metric_initstaticscaled_gilbertinit",
        # "pathfinder_metric_gilbertinit",
    ])
    idata[benchmark=At(benchmarks)]
end

# ╔═╡ 37cd1ca2-fa8b-4e41-a1cc-97f227646c4d
DataFrame(
    dropdims(
        mean(idata.sample_stats.diverging; dims=(:draw, :chain, :run));
        dims=(:draw, :chain, :run),
    ),
)

# ╔═╡ d385430b-0471-45de-b87b-a607ba2cd466
condition = rebuild(
    dropdims(
        mapslices(
            pdcond,
            idata.sample_stats.inv_metric;
            dims=(:inv_metric_dim_1, :inv_metric_dim_2),
        );
        dims=(:inv_metric_dim_1, :inv_metric_dim_2),
    );
    name=:condition_number,
)

# ╔═╡ 6c8a6cbd-b172-4e87-bdc5-0cd2ace061d3
let
    fig = draw(
        data(condition) *
        mapping(:condition_number => "condition number of final metric"; color=:benchmark) *
        visual(ECDFPlot);
        axis=merge(axis_ecdf, (; xscale=log10)),
    )
    # xlims!(; low=1, high=10^3)
    fig
end

# ╔═╡ d2ebe0fc-0dbc-4f0c-a95a-f58f255de712
rhat_vals = rebuild(
    stack(
        map(eachslice(idata.posterior.x; dims=(:run, :benchmark))) do xi
            xref = format_chains_as(ref_draws.x, xi)
            rhat(cat(xi, xref; dims=Dim{:chain}))
        end,
    ); name="rhat"
)

# ╔═╡ 819125a0-0a70-4f71-8a58-b825bf7c2e2d
let
    fig = draw(
        data(rhat_vals) * mapping(:rhat => L"\hat{R}"; color=:benchmark) * visual(ECDFPlot);
        axis=axis_ecdf,
    )
    vlines!(1.01; linestyle=:dash, color=:black)
    xlims!(; high=1.05)
    ylims!(; low=0.5, high=1.01)
    fig
end

# ╔═╡ 2adbcc5e-806f-4f6c-99f3-8d024f8e9807
converg_diag = let
    x = idata.posterior.x
    xreshaped = reshape(x, size(x, :draw), size(x, :chain), :)
    S, R = ess_rhat(xreshaped)
    ess = DimArray(reshape(S, size(x)[3:end]), DimensionalData.dims(x)[3:end])
    rhat = DimArray(reshape(R, size(x)[3:end]), DimensionalData.dims(x)[3:end])
    convert_to_dataset((; ess, rhat))
end

# ╔═╡ 3ed64104-147b-4f81-9082-1653b91cad09
let
    fig = draw(
        data(converg_diag) * mapping(:ess => "ESS"; color=:benchmark) * visual(ECDFPlot);
        axis=merge(axis_ecdf, (; xscale=log10)),
    )
    # xlims!(; low=10^4)
    fig
end

# ╔═╡ 3b355df6-5a7e-41c4-9bcf-db7aaf6a734b
ess_per = let
    num_evals = dropdims(
        sum(idata.sample_stats.num_evals; dims=(:chain, :eval_type));
        dims=(:chain, :eval_type),
    )
    num_evals_total = dropdims(sum(num_evals; dims=:stage); dims=:stage)
    time = dropdims(sum(idata.sample_stats.time; dims=:chain); dims=:chain)
    ess_per_nevals = broadcast_dims(/, converg_diag.ess, num_evals)
    ess_per_nevals_total = broadcast_dims(/, converg_diag.ess, num_evals_total)
    ess_per_sec = broadcast_dims(/, converg_diag.ess, time)
    ess_per_sec_total = broadcast_dims(
        /, converg_diag.ess, dropdims(sum(time; dims=:stage); dims=:stage)
    )
    convert_to_dataset((;
        ess_per_nevals, ess_per_nevals_total, ess_per_sec, ess_per_sec_total
    ))
end

# ╔═╡ 44fb7602-a32d-48ab-8658-73eab132b1f4
dropdims(minimum(ess_per.ess_per_nevals_total; dims=:param); dims=:param)

# ╔═╡ c90a0c05-a942-446f-a7c7-726b7d6e11da
let
    d = dropdims(minimum(ess_per; dims=:param); dims=:param)
    fig = draw(
        data(d.ess_per_nevals_total) *
        mapping(
            :ess_per_nevals_total => :min_ess_per_nevals_total,
            :benchmark;
            group=:run => nonnumeric,
        ) *
        visual(Lines; linewidth=0.2);
        axis=(; xscale=log10),
    )
    # xlims!(; low=10^-2)
    fig
end

# ╔═╡ 2525e03a-1792-4693-a6e3-4003f74e7344
draw(
    data(minimum(ess_per.ess_per_nevals; dims=:param)) *
    mapping(:ess_per_nevals => "ESS/(# evaluations)"; color=:benchmark, col=:stage) *
    visual(ECDFPlot);
    facet=(; linkxaxes=:none),
    axis=axis_ecdf,
)

# ╔═╡ be7bcb5c-3769-4c8e-8283-169808d5afe6
let
    fig = draw(
        data(minimum(ess_per.ess_per_nevals_total; dims=:param)) *
        mapping(:ess_per_nevals_total => "ESS/(total # evaluations)"; color=:benchmark) *
        visual(ECDFPlot);
        axis=merge(axis_ecdf, (xscale=log10,)),
    )
    # xlims!(; low=10^-1)
    fig
end

# ╔═╡ bbe121fb-5ed6-4e1e-ad84-7b9c9627ece9
draw(
    data(minimum(ess_per.ess_per_sec; dims=:param)) *
    mapping(:ess_per_sec => "ESS/runtime (s⁻¹)"; color=:benchmark, col=:stage) *
    visual(ECDFPlot);
    facet=(; linkxaxes=:none),
    axis=merge(axis_ecdf, (; xscale=log10)),
)

# ╔═╡ 0794c8ae-3d81-4c62-8998-3bdbe217601d
let
    fig = draw(
        data(minimum(ess_per; dims=:param)) *
        mapping(:ess_per_sec_total => "ESS/(total runtime) (s⁻¹)"; color=:benchmark) *
        visual(ECDFPlot);
        axis=merge(axis_ecdf, (; xscale=log10)),
    )
    # xlims!(; low=10^3)
    fig
end

# ╔═╡ Cell order:
# ╠═a791fbea-a7e4-11ed-0c1e-0f45961d3d26
# ╠═6ab900d1-008c-4cde-b114-408c1adcea33
# ╠═f7084e2e-196e-41c9-b791-28fff8c4b0ae
# ╠═b45378f6-5a1c-4117-a2ee-0b8809283bec
# ╠═144af9d2-a6de-4c8c-8feb-bbe61d6c28e8
# ╠═2dab771d-2d8e-4603-8aa9-7a7f02e98b41
# ╠═88f26eb3-8ca5-4146-aa66-8ae9ecd440a9
# ╠═37cd1ca2-fa8b-4e41-a1cc-97f227646c4d
# ╠═d385430b-0471-45de-b87b-a607ba2cd466
# ╠═6c8a6cbd-b172-4e87-bdc5-0cd2ace061d3
# ╠═d2ebe0fc-0dbc-4f0c-a95a-f58f255de712
# ╠═819125a0-0a70-4f71-8a58-b825bf7c2e2d
# ╠═2adbcc5e-806f-4f6c-99f3-8d024f8e9807
# ╠═3ed64104-147b-4f81-9082-1653b91cad09
# ╠═3b355df6-5a7e-41c4-9bcf-db7aaf6a734b
# ╠═44fb7602-a32d-48ab-8658-73eab132b1f4
# ╠═c90a0c05-a942-446f-a7c7-726b7d6e11da
# ╠═2525e03a-1792-4693-a6e3-4003f74e7344
# ╠═be7bcb5c-3769-4c8e-8283-169808d5afe6
# ╠═bbe121fb-5ed6-4e1e-ad84-7b9c9627ece9
# ╠═0794c8ae-3d81-4c62-8998-3bdbe217601d
