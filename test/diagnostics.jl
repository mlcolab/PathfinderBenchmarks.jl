using PathfinderPoster
using Test
using Pathfinder
using KrylovKit
using LinearAlgebra

Random.seed!(0)

@testset "diagnostics" begin
    @testset "svdvals_extreme" begin
        m = 10
        @testset "T=$T, n=$n" for T in (Float32, Float64), n in (5, 100)
            A = randn(T, n, n)
            @test all(PathfinderPoster.svdvals_extreme(A) .≈ extrema(svdvals(A)))
        end
    end

    @testset "pdcond" begin
        m = 10
        @testset "T=$T, n=$n" for T in (Float32, Float64), n in (5, 100)
            A = rand_pd_mat(T, n)
            B = randn(T, n, m)
            D = rand_pd_diag_mat(T, m)
            W = Pathfinder.WoodburyPDMat(A, B, D)
            @test pdcond(W) ≈ cond(W)
        end
    end
end
