using LinearAlgebra, LogDensityProblems, PathfinderBenchmarks, Test

struct IIDNormalProblem{M,order}
    μ::M
end
IIDNormalProblem(μ::M, order::Int) where {M} = IIDNormalProblem{M,order}(μ)

function LogDensityProblems.capabilities(::Type{IIDNormalProblem{M,O}}) where {M,O}
    return LogDensityProblems.LogDensityOrder{O}()
end
LogDensityProblems.dimension(prob::IIDNormalProblem) = length(prob.μ)
LogDensityProblems.logdensity(prob::IIDNormalProblem, x) = -sum(abs2, x - prob.μ) / 2
function LogDensityProblems.logdensity_and_gradient(prob::IIDNormalProblem, x)
    z = x - prob.μ
    return (-sum(abs2, z) / 2, -z)
end
function LogDensityProblems.logdensity_gradient_and_hessian(prob::IIDNormalProblem, x)
    z = x - prob.μ
    return (-sum(abs2, z) / 2, -z, diagm(map(x -> -one(x), z)))
end

@testset "EvalCountingProblem" begin
    @testset for dim in [5, 10], order in 0:2
        μ = randn(dim)
        prob = IIDNormalProblem(μ, order)
        prob_counting = PathfinderBenchmarks.EvalCountingProblem(prob)
        @test prob_counting isa PathfinderBenchmarks.EvalCountingProblem
        @test prob_counting.num_fun_evals == 0
        @test prob_counting.num_grad_evals == 0
        @test prob_counting.num_hess_evals == 0
        @test LogDensityProblems.dimension(prob_counting) ==
            LogDensityProblems.dimension(prob)
        @test LogDensityProblems.capabilities(prob_counting) ===
            LogDensityProblems.capabilities(prob)
        for i in 1:10
            x = randn(dim)
            lp = LogDensityProblems.logdensity(prob_counting, x)
            lp_exp = LogDensityProblems.logdensity(prob, x)
            @test lp == lp_exp
            @test prob_counting.num_fun_evals == i
            @test prob_counting.num_grad_evals == 0
            @test prob_counting.num_hess_evals == 0
        end
        order ≥ 1 || continue

        prob_counting = PathfinderBenchmarks.EvalCountingProblem(prob)
        for i in 1:10
            x = randn(dim)
            lp, g = LogDensityProblems.logdensity_and_gradient(prob_counting, x)
            lp_exp, g_exp = LogDensityProblems.logdensity_and_gradient(prob, x)
            @test lp == lp_exp
            @test g == g_exp
            @test prob_counting.num_fun_evals == 0
            @test prob_counting.num_grad_evals == i
            @test prob_counting.num_hess_evals == 0
        end
        order ≥ 2 || continue

        prob_counting = PathfinderBenchmarks.EvalCountingProblem(prob)
        for i in 1:10
            x = randn(dim)
            lp, g, h = LogDensityProblems.logdensity_gradient_and_hessian(prob_counting, x)
            lp_exp, g_exp, h_exp = LogDensityProblems.logdensity_gradient_and_hessian(
                prob, x
            )
            @test lp == lp_exp
            @test g == g_exp
            @test h == h_exp
            @test prob_counting.num_fun_evals == 0
            @test prob_counting.num_grad_evals == 0
            @test prob_counting.num_hess_evals == i
        end
    end
end
