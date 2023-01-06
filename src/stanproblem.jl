"""
    StanProblem(stan_file::String, data::String)

A struct representating a Stan model and implementing the LogDensityProblems interface.

`stan_file` is a path to a `.stan` file containing a model definition, while `data` is
either a path to a `.json` file containing data or a string containing data in JSON format.

The struct has the additional fields:
- `num_evals`: the number of times the log-density has been evaluated
- `num_grad_evals`: the number of times the gradient of the log-density has been evaluated

    StanProblem(post::PosteriorDB.Posterior)

Construct a `StanProblem` from the metadata in a `PosteriorDB.Posterior` object.
"""
mutable struct StanProblem{T}
    const model::T
    num_evals::Int
    num_grad_evals::Int
end
function StanProblem(stan_file::String, data::String)
    model = StanSample.BS.StanModel(; stan_file, data)
    return StanProblem(model, 0, 0)
end
function StanProblem(post::PosteriorDB.Posterior)
    model = PosteriorDB.model(post)
    stan_file = PosteriorDB.path(PosteriorDB.implementation(model, "stan"))
    # data is stored in a zipped JSON file, load as JSON string
    r = ZipFile.Reader(PosteriorDB.path(PosteriorDB.dataset(post)))
    data = read(only(r.files), String)
    close(r)
    return StanProblem(stan_file, data)
end

function Base.show(io::IO, ::MIME"text/plain", prob::StanProblem)
    return print(io, "StanProblem: $(StanSample.BS.name(prob.model))")
end

function LogDensityProblems.capabilities(::Type{<:StanProblem})
    return LogDensityProblems.LogDensityOrder{1}()  # can do gradient
end

function LogDensityProblems.dimension(prob::StanProblem)
    return Int(StanSample.BS.param_unc_num(prob.model))
end

function LogDensityProblems.logdensity(prob::StanProblem, x)
    model = prob.model
    lp = try
        StanSample.BS.log_density(model, convert(Vector{Float64}, x))
    catch
        NaN
    end
    prob.num_evals += 1
    return lp::Float64
end

function LogDensityProblems.logdensity_and_gradient(prob::StanProblem, x)
    m = prob.model
    lp_grad = try
        StanSample.BS.log_density_gradient(m, convert(Vector{Float64}, x))
    catch
        NaN, fill(NaN, length(x))
    end
    prob.num_grad_evals += 1
    return lp_grad::Tuple{Float64,Vector{Float64}}
end
