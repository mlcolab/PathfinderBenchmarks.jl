using Optim, OptimizationNLopt, Pathfinder, PathfinderBenchmarks, Test
using Optim: LineSearches

@testset "PathfinderConfig" begin
    cfg = PathfinderBenchmarks.PathfinderConfig()
    @test cfg.options === NamedTuple()

    optimizers = [LBFGS(), Newton()]
    @testset for optimizer in optimizers, history_length in [3, 6]
        cfg = PathfinderBenchmarks.PathfinderConfig(; optimizer, history_length)
        @test cfg.options === (; optimizer, history_length)
    end
end

@testset "all_pathfinder_configurations" begin
    cfgs = Dict(all_pathfinder_configurations(10))
    @testset for name in keys(cfgs)
        cfg = cfgs[name]
        indicators = split(name, '_')
        @test cfg isa PathfinderBenchmarks.PathfinderConfig
        if name == ""
            @test cfg === PathfinderBenchmarks.PathfinderConfig()
            continue
        end
        optimizer = cfg.options.optimizer
        if "nloptlbfgs" ∈ indicators
            @test optimizer isa NLopt.Opt
        else
            if "gilbertinit" ∈ indicators
                @test optimizer isa PathfinderBenchmarks.LBFGS
                @test optimizer.init_invH0 === PathfinderBenchmarks.init_invH0_gilbert!
            else
                @test optimizer isa Optim.LBFGS
            end
            @test optimizer.m == Pathfinder.DEFAULT_HISTORY_LENGTH
            if "backtrackingls" ∈ indicators
                @test optimizer.linesearch! isa LineSearches.BackTracking
            elseif "hagerzhangls" ∈ indicators
                @test optimizer.linesearch! isa LineSearches.HagerZhang
            else
                @test optimizer.linesearch! isa LineSearches.MoreThuente
            end
            if "initstaticscaled" ∈ indicators
                @test optimizer.alphaguess! isa LineSearches.InitialStatic
                @test optimizer.alphaguess!.scaled
            elseif "inithagerzhangls" ∈ indicators
                @test optimizer.alphaguess! isa LineSearches.InitialHagerZhang
            else
                @test optimizer.alphaguess! isa LineSearches.InitialStatic
                @test !optimizer.alphaguess!.scaled
            end
        end
    end
end
