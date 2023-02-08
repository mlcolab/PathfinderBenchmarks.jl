using Optim, Pathfinder, PathfinderBenchmarks, Test

@testset "PathfinderConfig" begin
    cfg = PathfinderBenchmarks.PathfinderConfig()
    optimizer = cfg.optimizer
    @test optimizer isa Optim.LBFGS
    @test optimizer.m == Pathfinder.DEFAULT_HISTORY_LENGTH
    @test optimizer.linesearch! isa Optim.LineSearches.MoreThuente

    optimizer = Newton()
    cfg2 = PathfinderBenchmarks.PathfinderConfig(optimizer)
    @test cfg2.optimizer === optimizer
end
