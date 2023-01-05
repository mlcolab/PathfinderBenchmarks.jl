using LinearAlgebra
using Random

function rand_pd_mat(rng, T, n)
    U = qr(randn(rng, T, n, n)).Q
    return Matrix(Symmetric(U * rand_pd_diag_mat(rng, T, n) * U'))
end
rand_pd_mat(T, n) = rand_pd_mat(Random.default_rng(), T, n)

rand_pd_diag_mat(rng, T, n) = Diagonal(rand(rng, T, n))
rand_pd_diag_mat(T, n) = rand_pd_diag_mat(Random.default_rng(), T, n)
