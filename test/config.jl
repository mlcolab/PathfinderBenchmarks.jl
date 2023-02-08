using Optim, PathfinderBenchmarks, Test

@testset "PathfinderConfig" begin
    cfg = PathfinderBenchmarks.PathfinderConfig()
    @test cfg.options === NamedTuple()

    optimizers = [LBFGS(), Newton()]
    @testset for optimizer in optimizers, history_length in [3, 6]
        cfg = PathfinderBenchmarks.PathfinderConfig(; optimizer, history_length)
        @test cfg.options === (; optimizer, history_length)
    end
end
