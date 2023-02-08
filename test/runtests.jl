using PathfinderBenchmarks
using Test

@testset "PathfinderBenchmarks.jl" begin
    include("test_utils.jl")
    include("diagnostics.jl")
    include("lbfgs.jl")
    include("counting_problem.jl")
    include("config.jl")
    include("hmc_initialization.jl")
end
