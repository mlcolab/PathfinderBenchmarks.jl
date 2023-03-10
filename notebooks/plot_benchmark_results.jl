### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a791fbea-a7e4-11ed-0c1e-0f45961d3d26
begin
    cd(@__DIR__)

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
    axis_ecdf = (
        width=500,
        height=450,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(10),
        ylabel="probability",
    )
end;

# ╔═╡ 2bb23963-eed9-411d-877e-b6ca712008c0
default_theme(Scatter)

# ╔═╡ 7c11dbeb-6474-47f2-b1bb-418f455fc85a
hmc_benchmarks_defaults = sort([
    "default_dense",
    "default_diag",
    "pathfinder_metric",
    "pathfinder_metric_diag_init",
    "pathfinder_metric_dense_init",
]);

# ╔═╡ 41bcf76e-a914-421c-a994-3ea70bf5c3e5
hmc_benchmarks_variants = sort([
    "default_dense",
    "default_diag",
    "pathfinder_metric",
    "pathfinder_metric_hagerzhangls",
    "pathfinder_metric_inithagerzhangls_hagerzhangls",
    "pathfinder_metric_initstaticscaled",
    "pathfinder_metric_initstaticscaled_gilbertinit",
]);

# ╔═╡ 0dd9779a-4b3d-4d75-a061-226677a13988
pf_benchmarks = sort([
    "default",
    "hagerzhangls",
    "inithagerzhangls_hagerzhangls",
    "initstaticscaled",
    "initstaticscaled_gilbertinit",
]);

# ╔═╡ 1a8972c6-9715-4c68-86c8-8bb1a9c34bfd
model_name(posterior_name) = split(string(posterior_name), '-')[2];

# ╔═╡ 6ea607cd-6285-449d-a719-cb708ba5a100
let
    pf_benchmark_parts = map(x -> split(x, '_'), pf_benchmarks)
    ls = map(pf_benchmark_parts) do x
        i = findfirst(∈(("hagerzhangls", "backtrackingls")), x)
        i === nothing && return "morethuente"
        return first(split(x[i], "ls"))
    end
    lsinit = map(pf_benchmark_parts) do x
        i = findfirst(∈(("initstaticscaled", "inithagerzhangls")), x)
        i === nothing && return "static"
        return first(split(last(split(x[i], "init")), "ls"))
    end
    invH0init = map(pf_benchmark_parts) do x
        return "gilbertinit" ∈ x ? "gilbert" : "nocedalwright"
    end
    benchmark_dim = Dim{:benchmark}(pf_benchmarks)
    ds_benchmark = Dataset((;
        linesearch=DimArray(ls, benchmark_dim),
        linesearch_init=DimArray(lsinit, benchmark_dim),
        invH0_init=DimArray(invH0init, benchmark_dim),
    ))
    ds = merge(hess_diagnostics, ds_benchmark)
    fig = draw(
        (
            (
                data(ds[(:cond_pathfinder, :linesearch, :linesearch_init, :invH0_init)]) *
                mapping(:cond_pathfinder; color=:benchmark, col=:cond)
            ) +
            (data(ds.cond_Σref) * mapping(:cond_Σref; col=:cond)) *
            visual(; color=:black, linestyle=:dash)
            # (
            # 	data(hess_diagnostics.cond_ref) *
            #        	mapping(:cond_ref; col=:cond)
            # ) * visual(; color=:gray, linestyle=:dash)
        ) * visual(ECDFPlot);
        axis=merge(axis_ecdf, (; xscale=log10)),
    )
    # xlims!(; low=1, high=10^3)
    fig
end

# ╔═╡ f2abba93-e887-4e29-8fdc-b2b471a1d4f4
posterior_names = [
    "arma-arma11", "diamonds-diamonds", "eight_schools-eight_schools_centered"
];

# ╔═╡ a04e8a69-6cdf-4751-8106-3ea11450d2da
results = (; map(posterior_names) do name
    return Symbol(name) => from_netcdf("$name/benchmark_summary.nc")
end...)

# ╔═╡ 5a41abb8-4852-43f5-92a8-87226c9d1b38
md"""
## HMC benchmark results with defaults
"""

# ╔═╡ d26ccaee-fa84-40c6-bc94-908b1ba399c0
let fig = Figure(; resolution=(900, 800))
    for (i, (name, result)) in enumerate(pairs(results))
        ax = Axis(
            fig[1, i];
            title="$(model_name(name))",
            xlabel=i == 2 ? L"\hat{R}" : "",
            ylabel="probability",
        )
        convergence_diag = view(
            result.hmc_benchmark_stats.convergence_diag;
            _metric=At("rhat_vs_ref"),
            benchmark=At(hmc_benchmarks_defaults),
        )
        layers =
            data(convergence_diag) *
            mapping(:convergence_diag; color=:benchmark) *
            visual(ECDFPlot; alpha=0.9)
        draw!(ax, layers)
        vlines!(ax, 1.01; color=:black, linestyle=:dash)
        xlims!(ax, 0.999, 1.02)
        if i > 1
            hideydecorations!(ax)
            hidespines!(ax, :l)
        end
    end
    xlims = (; var"arma-arma11"=(10^(-1.4), 10^-0.5))
    xscales = (; var"diamonds-diamonds"=log10)
    lin = nothing
    for (i, (name, result)) in enumerate(pairs(results))
        ax = Axis(
            fig[2, i];
            xlabel=i == 2 ? "ESS/nevals" : "",
            ylabel="probability",
            xscale=get(xscales, name, identity),
            titlesize=14,
        )
        performance = view(
            result.hmc_benchmark_stats.performance;
            _metric=At("ess_bulk"),
            per=At("num_evals"),
            benchmark=At(hmc_benchmarks_defaults),
        )
        layers =
            data(performance) *
            mapping(:performance; color=:benchmark) *
            visual(ECDFPlot; alpha=0.9)
        lin = draw!(ax, layers)
        name ∈ keys(xlims) && xlims!(ax, xlims[name]...)
        ylims!(ax, -0.01, 1.01)
        if i > 1
            hideydecorations!(ax)
            hidespines!(ax, :l)
        end
    end
    legend!(fig[3, 1:3], lin; orientation=:horizontal, nbanks=2, titlesize=0)
    Makie.save("hmc_default_convergence_performance.pdf", fig)
    fig
end

# ╔═╡ 19d3555b-2aef-4146-b11c-6ad076cb1499
md"""
## HMC benchmark results with variants
"""

# ╔═╡ 1fef6416-1f2a-41a5-bfd9-2f6371e546d2
let fig = Figure(; resolution=(900, 820))
    for (i, (name, result)) in enumerate(pairs(results))
        ax = Axis(
            fig[1, i];
            title="$(model_name(name))",
            xlabel=i == 2 ? L"\hat{R}" : "",
            ylabel="probability",
        )
        convergence_diag = view(
            result.hmc_benchmark_stats.convergence_diag;
            _metric=At("rhat_vs_ref"),
            benchmark=At(hmc_benchmarks_variants),
        )
        layers =
            data(convergence_diag) *
            mapping(:convergence_diag; color=:benchmark) *
            visual(ECDFPlot; alpha=0.9)
        draw!(ax, layers)
        vlines!(ax, 1.01; color=:black, linestyle=:dash)
        xlims!(ax, 0.999, 1.02)
        if i > 1
            hideydecorations!(ax)
            hidespines!(ax, :l)
        end
    end
    xlims = (; var"arma-arma11"=(10^(-1.4), 10^-0.5))
    xscales = (; var"diamonds-diamonds"=log10)
    lin = nothing
    for (i, (name, result)) in enumerate(pairs(results))
        ax = Axis(
            fig[2, i];
            xlabel=i == 2 ? "ESS/nevals" : "",
            ylabel="probability",
            xscale=get(xscales, name, identity),
            titlesize=14,
        )
        performance = view(
            result.hmc_benchmark_stats.performance;
            _metric=At("ess_bulk"),
            per=At("num_evals"),
            benchmark=At(hmc_benchmarks_variants),
        )
        layers =
            data(performance) *
            mapping(:performance; color=:benchmark) *
            visual(ECDFPlot; alpha=0.9)
        lin = draw!(ax, layers)
        name ∈ keys(xlims) && xlims!(ax, xlims[name]...)
        ylims!(ax, -0.01, 1.01)
        if i > 1
            hideydecorations!(ax)
            hidespines!(ax, :l)
        end
    end
    legend!(fig[3, 1:3], lin; orientation=:horizontal, nbanks=4, titlesize=0)
    Makie.save("hmc_variants_convergence_performance.pdf", fig)
    fig
end

# ╔═╡ 29974d45-b5dd-4b37-87b0-50f57ac3756d
md"""
## Diagnostics
"""

# ╔═╡ 969c20d2-5f3d-4bfa-97e6-19a8f6870072
md"""
### Condition numbers
"""

# ╔═╡ d99e5315-8c2b-4b06-91d3-6a0c5fc51ce5
let fig = Figure(; resolution=(1000, 300))
    lin = nothing
    for (i, (name, result)) in enumerate(pairs(results))
        cov_ref = result.reference_posterior_stats.cov_unc
        cond_ref = rebuild(
            DimArray(
                [
                    cond(cov_ref),
                    cond(Diagonal(cov_ref)),
                    cond(cov_ref) / cond(Diagonal(cov_ref)),
                ],
                Dimensions.dims(result.reference_posterior_stats, :cond),
            );
            name=:cond_ref,
        )
        condition_inv_hessian = result.pathfinder_benchmark_stats.condition_inv_hessian[benchmark=At(
            "inithagerzhangls_hagerzhangls"
        )]

        ax = Axis(
            fig[1, i];
            title="$(model_name(name))",
            xlabel=i == 2 ? "condition number of approximate inverse Hessian" : "",
            ylabel="probability",
            xscale=log10,
        )

        layers = (
            data(condition_inv_hessian) *
            mapping(:condition_inv_hessian; color=:cond) *
            visual(ECDFPlot; alpha=0.9) +
            data(cond_ref) *
            mapping(:cond_ref; color=:cond) *
            visual(ECDFPlot; linestyle=:dash, alpha=0.5)
        )

        lin = draw!(ax, layers)
        if i > 1
            hideydecorations!(ax)
            hidespines!(ax, :l)
        end
    end
    legend!(fig[1, 4], lin; titlesize=0)
    Makie.save("pathfinder_condition_number.pdf", fig)
    fig
end

# ╔═╡ 40bf19aa-792e-49ee-87d4-189e9ad2ecb1
md"""
### Number of BFGS updates rejected
"""

# ╔═╡ a2d5690e-156c-47eb-a672-fc0cd40f0446
let fig = Figure()
    num_bfgs_updates_rejected = map(results) do result
        return result.pathfinder_results.num_bfgs_updates_rejected[benchmark=At(
            "inithagerzhangls_hagerzhangls"
        )]
    end
    all_data = sum(keys(num_bfgs_updates_rejected)) do k
        data(cat(num_bfgs_updates_rejected[k]; dims=Dim{:model}([model_name(k)])))
    end
    fig = Figure(; resolution=(700, 450))
    ax = Axis(fig[1, 1]; xlabel="number of L-BFGS updates rejected", ylabel="probability")

    grid = draw!(
        ax,
        all_data *
        mapping(
            :num_bfgs_updates_rejected => "number of L-BFGS updates rejected"; color=:model
        ) *
        visual(ECDFPlot; alpha=0.9),
    )
    AlgebraOfGraphics.legend!(
        fig[1, 1],
        grid;
        tellwidth=false,
        tellheight=false,
        valign=:bottom,
        halign=:right,
        padding=(0, 0, 20, 0),
        titlesize=0,
    )
    xlims!(-5, 202)
    ylims!(-0.01, 1.01)
    Makie.save("num_bfgs_updates_rejected.pdf", fig)
    fig
end

# ╔═╡ ba4dd852-fd57-4018-8578-6ff1afdd2d57
function pairplot(data::AbstractDimArray; dims, visual=Scatter, kwargs...)
    nrows = ncols = size(data, dims) - 1
    inds = DimensionalData.DimIndices(Dimensions.dims(data, dims))
    lookup = DimensionalData.lookup(data, dims)
    f = Figure(; kwargs...)
    for i in 1:(nrows + 1), j in 1:(i - 1)
        xi = view(data, inds[i]...)
        xj = view(data, inds[j]...)
        xlabel = "$(lookup[i])"
        ylabel = "$(lookup[j])"
        plot(visual, f[i - 1, j], vec(xi), vec(xj); axis=(; xlabel, ylabel))
    end
    return f
end

# ╔═╡ 04b5a903-c169-41b6-9387-7a859e7ccb76
# ╠═╡ disabled = true
#=╠═╡
fig = let
	axis=(width=200, height=200)
	x_unc = ref_draws.x_unc[param_unc=At(["b.2", "b.3", "b.22", "b.23"])]
	x_unc2 = set(x_unc; param_unc=:param_unc2)
	ds = Dataset((; x_unc, x_unc2))
	fig = draw(data(ds) * mapping(:x_unc => "", :x_unc2 => ""; col=:param_unc, row=:param_unc2) * visual(Hexbin; bins=50); facet = (; linkxaxes=:minimal, linkyaxes = :minimal), axis)
	grid = fig.grid
	lookup = sort(DimensionalData.lookup(x_unc, :param_unc))
	for i in axes(grid, 1), j in axes(grid, 2)
		if i ≥ j
			delete!(grid[i, j].axis)
		end
		if i == j
			ax = Makie.density(fig.figure[i, j], vec(x_unc[param_unc=At(lookup[i])]); color = (:black, 0), strokewidth=2, axis)
		end
	end
	# for i in axes(grid, 1)
	# 	i == size(grid, 1) && continue
	# 	linkyaxes!(map(g -> g.axis, grid[i, (i+1):end])...)
	# end
	# for j in axes(grid, 2)
	# 	# j == size(grid, 2) && continue
	# 	linkxaxes!(map(g -> g.axis, grid[:, j])...)
	# end
	for g in fig.grid
		hidedecorations!(g.axis)
		hidespines!(g.axis)
	end
	fig
end
  ╠═╡ =#

# ╔═╡ 6c8a6cbd-b172-4e87-bdc5-0cd2ace061d3
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═a791fbea-a7e4-11ed-0c1e-0f45961d3d26
# ╠═6ab900d1-008c-4cde-b114-408c1adcea33
# ╠═2bb23963-eed9-411d-877e-b6ca712008c0
# ╠═7c11dbeb-6474-47f2-b1bb-418f455fc85a
# ╠═41bcf76e-a914-421c-a994-3ea70bf5c3e5
# ╠═0dd9779a-4b3d-4d75-a061-226677a13988
# ╠═1a8972c6-9715-4c68-86c8-8bb1a9c34bfd
# ╠═6ea607cd-6285-449d-a719-cb708ba5a100
# ╠═f2abba93-e887-4e29-8fdc-b2b471a1d4f4
# ╠═a04e8a69-6cdf-4751-8106-3ea11450d2da
# ╟─5a41abb8-4852-43f5-92a8-87226c9d1b38
# ╠═d26ccaee-fa84-40c6-bc94-908b1ba399c0
# ╟─19d3555b-2aef-4146-b11c-6ad076cb1499
# ╠═1fef6416-1f2a-41a5-bfd9-2f6371e546d2
# ╟─29974d45-b5dd-4b37-87b0-50f57ac3756d
# ╟─969c20d2-5f3d-4bfa-97e6-19a8f6870072
# ╠═d99e5315-8c2b-4b06-91d3-6a0c5fc51ce5
# ╟─40bf19aa-792e-49ee-87d4-189e9ad2ecb1
# ╠═a2d5690e-156c-47eb-a672-fc0cd40f0446
# ╠═ba4dd852-fd57-4018-8578-6ff1afdd2d57
# ╠═04b5a903-c169-41b6-9387-7a859e7ccb76
# ╠═6c8a6cbd-b172-4e87-bdc5-0cd2ace061d3
