using LinearAlgebra,
    LogDensityProblems,
    Optim,
    Pathfinder,
    PathfinderBenchmarks,
    PosteriorDB,
    Random,
    StanLogDensityProblems,
    Test

@testset "LBFGS" begin
    model_path = mktempdir(; cleanup=false)
    pdb = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, "dogs-dogs")
    prob = StanProblem(post, model_path; nan_on_error=true)
    dim = LogDensityProblems.dimension(prob)
    optim_fun = Pathfinder.build_optim_function(prob)

    @testset "consistency with Pathfinder's inverse hessians" begin
        @testset for history_length in [3, 6, 10],
            init_invH0 in [
                PathfinderBenchmarks.init_invH0_nocedal_wright!,
                PathfinderBenchmarks.init_invH0_gilbert!,
            ]

            x₀ = rand(dim) .* 4 .- 2
            optim_prob = Pathfinder.build_optim_problem(optim_fun, x₀)
            optimizer = PathfinderBenchmarks.LBFGS(; m=history_length, init_invH0)

            sol, optim_trace = Pathfinder.optimize_with_trace(optim_prob, optimizer)

            # run lbfgs_inverse_hessians with the same initialization as Optim.LBFGS
            Hs, num_bfgs_updates_rejected = Pathfinder.lbfgs_inverse_hessians(
                optim_trace.points,
                optim_trace.gradients;
                history_length,
                Hinit=(x, s, y) -> init_invH0(copy(x), s, y),
            )
            ss = diff(optim_trace.points)
            ps = (Hs .* optim_trace.gradients)[1:(end - 1)]

            # check that next direction computed from Hessian is the same as the actual
            # direction that was taken
            @test all(≈(1), dot.(ps, ss) ./ norm.(ss) ./ norm.(ps))
            @test num_bfgs_updates_rejected == 0
        end
    end

    @testset "consistency with Optim.LBFGS" begin
        @testset for history_length in [3, 6, 10]
            seed = Random.rand(UInt16)
            optimizer = Optim.LBFGS(; m=history_length)
            rng = Random.seed!(seed)
            result1 = pathfinder(prob; rng, optimizer)

            optimizer = PathfinderBenchmarks.LBFGS(;
                m=history_length, init_invH0=PathfinderBenchmarks.init_invH0_nocedal_wright!
            )
            rng = Random.seed!(seed)
            result2 = pathfinder(prob; rng, optimizer)

            @test result1.draws == result2.draws
            @test result1.fit_distribution.μ == result2.fit_distribution.μ
            @test result1.fit_distribution.Σ == result2.fit_distribution.Σ
        end
    end
end
