using PathfinderPoster
using Test

@testset "PathfinderPoster.jl" begin
    include("test_utils.jl")
    include("diagnostics.jl")
    include("stanproblem.jl")
end
