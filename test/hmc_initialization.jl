using DynamicHMC,
    LinearAlgebra,
    LogDensityProblems,
    Optim,
    Pathfinder,
    PathfinderBenchmarks,
    PosteriorDB,
    Random,
    StanLogDensityProblems,
    Test

@testset "HMC initialization" begin
    model_path = mktempdir(; cleanup=false)
    pdb = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, "dogs-dogs")
    prob = StanProblem(post, model_path; nan_on_error=true)
    dim = LogDensityProblems.dimension(prob)
    @testset for T in [PathfinderPointInitialization, PathfinderPointMetricInitialization]
        @testset for ϵ in [nothing, 0.01], optimizer in [LBFGS(), Newton()]
            q = randn(dim)
            κ = GaussianKineticEnergy(Diagonal([1.0, 2.0, 3.0]))
            initialization = (; q, ϵ, κ)
            warmup_stages = (T(PathfinderConfig(; optimizer)),)

            seed = rand(UInt16)
            rng = MersenneTwister(seed)
            result = DynamicHMC.mcmc_keep_warmup(
                rng, prob, 0; warmup_stages, initialization
            )

            rng = MersenneTwister(seed)
            pf_result = pathfinder(prob; ndraws=1, rng, optimizer, init=q)
            q′ = pf_result.draws[:, 1]

            @test result.initial_warmup_state.Q.q == q
            @test result.final_warmup_state.Q.q == q′
            (; ℓq, ∇ℓq) = result.final_warmup_state.Q
            @test (ℓq, ∇ℓq) == LogDensityProblems.logdensity_and_gradient(prob, q′)
            @test result.final_warmup_state.ϵ === ϵ

            if T === PathfinderPointInitialization
                @test result.final_warmup_state.κ === κ
            else
                @test result.final_warmup_state.κ.M⁻¹ isa Pathfinder.WoodburyPDMat
                @test result.final_warmup_state.κ.M⁻¹ == pf_result.fit_distribution.Σ
            end
        end
    end
end
