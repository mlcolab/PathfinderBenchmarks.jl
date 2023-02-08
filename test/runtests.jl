using PathfinderBenchmarks
using Test

@testset "PathfinderBenchmarks.jl" begin
    include("test_utils.jl")
    include("config.jl")
    include("diagnostics.jl")
    include("lbfgs.jl")
end
