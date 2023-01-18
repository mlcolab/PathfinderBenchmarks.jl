using PathfinderBenchmarks
using Test
using JSON
using LogDensityProblems
using PosteriorDB
using StanSample

regression_model_code = """
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + beta * x, sigma);
}
"""

@testset "StanProblem" begin
    if !isdefined(StanSample, :BS)
        @warn "BridgeStan not installed, skipping StanProblem tests"
        return nothing
    end
    @testset "StanProblem(stan_file, data)" begin
        mktempdir() do path
            stan_file = joinpath(path, "mymodel.stan")
            write(stan_file, regression_model_code)
            N = 10
            x = randn(N)
            y = randn(N)
            data = (; N, x, y)
            data_json = JSON.json(data)
            prob = StanProblem(stan_file, data_json)
            @test prob isa StanProblem
            @test prob.model isa StanSample.BS.StanModel
            @test prob.num_evals == 0
            @test prob.num_grad_evals == 0
            @test sprint(show, "text/plain", prob) == "StanProblem: mymodel_model"

            @test LogDensityProblems.dimension(prob) == 3
            @test LogDensityProblems.capabilities(prob) ===
                LogDensityProblems.LogDensityOrder{1}()
            θ = randn(3)
            lp = @inferred LogDensityProblems.logdensity(prob, θ)
            @test lp isa Float64
            @test prob.num_evals == 1
            @test prob.num_grad_evals == 0

            lp = @inferred LogDensityProblems.logdensity(prob, fill(NaN, 3))
            @test isnan(lp)
            @test prob.num_evals == 2

            θ = randn(3)
            lp, grad = @inferred LogDensityProblems.logdensity_and_gradient(prob, θ)
            @test lp isa Float64
            @test grad isa Vector{Float64}
            @test prob.num_evals == 2
            @test prob.num_grad_evals == 1

            lp, grad = @inferred LogDensityProblems.logdensity_and_gradient(
                prob, fill(NaN, 3)
            )
            @test isnan(lp)
            @test all(isnan, grad)
            @test prob.num_evals == 2
            @test prob.num_grad_evals == 2
        end
    end

    @testset "StanProblem(::Posterior)" begin
        post = PosteriorDB.posterior(PosteriorDB.database(), "dogs-dogs")
        prob = StanProblem(post)
        @test prob isa StanProblem
        @test prob.model isa StanSample.BS.StanModel
        @test prob.num_evals == 0
        @test prob.num_grad_evals == 0
        θ = randn(LogDensityProblems.dimension(prob))
        LogDensityProblems.logdensity(prob, θ)
        LogDensityProblems.logdensity_and_gradient(prob, θ)
    end
end
