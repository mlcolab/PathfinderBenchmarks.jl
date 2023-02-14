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
        PathfinderBenchmarks,
        Statistics
    using AlgebraOfGraphics: density
end

# ╔═╡ 6ab900d1-008c-4cde-b114-408c1adcea33
begin
    CairoMakie.activate!(; type="svg")
    set_theme!(theme_minimal())
    update_theme!(; fontsize=18, font="CMU", linewidth=2)
    axis = (
        width=500, height=450, xminorticksvisible=true, xminorticks=IntervalsBetween(10)
    )
end;

# ╔═╡ 8b726bc5-685d-40de-b27d-4a0e980b1bab
# TODO: remove
function Base.cat(data::InferenceData, others::InferenceData...; groups=keys(data), dims)
    groups_cat = map(groups) do k
        k => cat(data[k], (other[k] for other in others)...; dims=dims)
    end
    # keep other non-concatenated groups
    return merge(data, others..., InferenceData(; groups_cat...))
end

# ╔═╡ 88f26eb3-8ca5-4146-aa66-8ae9ecd440a9
idata = let
    model_path = "eight_schools-eight_schools_centered"
    results_file = joinpath(model_path, "all_results.nc")
    if isfile(results_file)
        from_netcdf(results_file)
    else
        subdirs = filter(startswith("results_"), readdir(model_path))
        run_idatas = map(subdirs) do subdir
            name = subdir[9:end]
            path = joinpath(model_path, subdir)
            idatas = map(from_netcdf, readdir(path; join=true))
            return name => cat(idatas...; dims=:run)
        end
        idata = cat(map(last, run_idatas)...; dims=Dim{:benchmark}(map(first, run_idatas)))
        to_netcdf(idata, results_file)
        idata
    end
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
        data(condition) * mapping(:condition_number; color=:benchmark) * visual(ECDFPlot);
        axis=merge(axis, (; xscale=log10)),
    )
    fig
end

# ╔═╡ 2adbcc5e-806f-4f6c-99f3-8d024f8e9807
converg_diag = let
    x = idata.posterior.x
    xreshaped = reshape(x, size(x, :draw), size(x, :chain), :)
    val = ess_rhat(xreshaped)
    ess = DimArray(reshape(val.ess, size(x)[3:end]), DimensionalData.dims(x)[3:end])
    rhat = DimArray(reshape(val.rhat, size(x)[3:end]), DimensionalData.dims(x)[3:end])
    convert_to_dataset((; ess, rhat))
end

# ╔═╡ 3ed64104-147b-4f81-9082-1653b91cad09
draw(
    data(converg_diag) * mapping(:ess; color=:benchmark) * visual(ECDFPlot);
    axis=merge(axis, (; xscale=log10)),
)

# ╔═╡ 3b355df6-5a7e-41c4-9bcf-db7aaf6a734b
ess_per = let
    num_evals = dropdims(sum(idata.sample_stats.num_evals; dims=:chain); dims=:chain)
    num_evals_total = dropdims(
        sum(num_evals; dims=(:stage, :eval_type)); dims=(:stage, :eval_type)
    )
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

# ╔═╡ c90a0c05-a942-446f-a7c7-726b7d6e11da
let
    d = dropdims(minimum(ess_per; dims=:param); dims=:param)
    draw(
        data(d.ess_per_nevals_total) *
        mapping(
            :ess_per_nevals_total => :min_ess_per_nevals_total,
            :benchmark;
            group=:run => nonnumeric,
        ) *
        visual(Lines; linewidth=0.2);
        axis=merge(axis, (; xscale=log10)),
    )
end

# ╔═╡ 2525e03a-1792-4693-a6e3-4003f74e7344
draw(
    data(minimum(ess_per[eval_type=At(["grad"])]; dims=:param)) *
    mapping(:ess_per_nevals => :ess_per_ngrad_evals; color=:benchmark, col=:stage) *
    visual(ECDFPlot);
    facet=(; linkxaxes=:none),
    axis=merge(axis, (; xscale=log10)),
)

# ╔═╡ be7bcb5c-3769-4c8e-8283-169808d5afe6
let
    x = broadcast_dims(
        /,
        ess_per.ess_per_nevals_total,
        ess_per.ess_per_nevals_total[benchmark=At("default_diag")],
    )
    draw(
        data(minimum(x; dims=:param)) *
        mapping(:ess_per_nevals_total; color=:benchmark) *
        visual(ECDFPlot);
        axis=merge(axis, (; xscale=log10)),
    )
end

# ╔═╡ 8de8cd7f-d8c9-466d-8e0f-1536f822c178
broadcast_dims(
    /,
    ess_per.ess_per_nevals_total,
    ess_per.ess_per_nevals_total[benchmark=At("default_diag")],
)

# ╔═╡ ab4ffad1-7bcc-4d54-a80b-ff0243ed6b50
ess_per

# ╔═╡ bbe121fb-5ed6-4e1e-ad84-7b9c9627ece9
draw(
    data(minimum(ess_per.ess_per_sec; dims=:param)) *
    mapping(:ess_per_sec; color=:benchmark, col=:stage) *
    visual(ECDFPlot);
    facet=(; linkxaxes=:none),
    axis=merge(axis, (; xscale=log10)),
)

# ╔═╡ 0794c8ae-3d81-4c62-8998-3bdbe217601d
draw(
    data(minimum(ess_per; dims=:param)) *
    mapping(:ess_per_sec_total; color=:benchmark) *
    visual(ECDFPlot);
    axis=merge(axis, (; xscale=log10)),
)

# ╔═╡ Cell order:
# ╠═a791fbea-a7e4-11ed-0c1e-0f45961d3d26
# ╠═6ab900d1-008c-4cde-b114-408c1adcea33
# ╠═8b726bc5-685d-40de-b27d-4a0e980b1bab
# ╠═88f26eb3-8ca5-4146-aa66-8ae9ecd440a9
# ╠═37cd1ca2-fa8b-4e41-a1cc-97f227646c4d
# ╠═d385430b-0471-45de-b87b-a607ba2cd466
# ╠═6c8a6cbd-b172-4e87-bdc5-0cd2ace061d3
# ╠═2adbcc5e-806f-4f6c-99f3-8d024f8e9807
# ╠═3ed64104-147b-4f81-9082-1653b91cad09
# ╠═3b355df6-5a7e-41c4-9bcf-db7aaf6a734b
# ╠═c90a0c05-a942-446f-a7c7-726b7d6e11da
# ╠═2525e03a-1792-4693-a6e3-4003f74e7344
# ╠═be7bcb5c-3769-4c8e-8283-169808d5afe6
# ╠═8de8cd7f-d8c9-466d-8e0f-1536f822c178
# ╠═ab4ffad1-7bcc-4d54-a80b-ff0243ed6b50
# ╠═bbe121fb-5ed6-4e1e-ad84-7b9c9627ece9
# ╠═0794c8ae-3d81-4c62-8998-3bdbe217601d
