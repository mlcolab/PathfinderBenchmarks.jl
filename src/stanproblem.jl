"""
    StanProblem(stan_file::String, data::String)

A struct representating a Stan model and implementing the LogDensityProblems interface.

`stan_file` is a path to a `.stan` file containing a model definition, while `data` is
either a path to a `.json` file containing data or a string containing data in JSON format.

    StanProblem(post::PosteriorDB.Posterior)

Construct a `StanProblem` from the metadata in a `PosteriorDB.Posterior` object.
"""
struct StanProblem{T}
    model::T
end
function StanProblem(stan_file::String, data::String)
    model = BridgeStan.StanModel(; stan_file, data)
    return StanProblem(model)
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
    return print(io, "StanProblem: $(BridgeStan.name(prob.model))")
end

function LogDensityProblems.capabilities(::Type{<:StanProblem})
    return LogDensityProblems.LogDensityOrder{1}()  # can do gradient
end

function LogDensityProblems.dimension(prob::StanProblem)
    return Int(BridgeStan.param_unc_num(prob.model))
end

function LogDensityProblems.logdensity(prob::StanProblem, x)
    model = prob.model
    lp = try
        BridgeStan.log_density(model, convert(Vector{Float64}, x))
    catch
        NaN
    end
    return lp::Float64
end

function LogDensityProblems.logdensity_and_gradient(prob::StanProblem, x)
    m = prob.model
    lp_grad = try
        BridgeStan.log_density_gradient(m, convert(Vector{Float64}, x))
    catch
        NaN, fill(NaN, length(x))
    end
    return lp_grad::Tuple{Float64,Vector{Float64}}
end

function constrain(prob::StanProblem, x::AbstractVector)
    return BridgeStan.param_constrain(prob.model, x)
end
