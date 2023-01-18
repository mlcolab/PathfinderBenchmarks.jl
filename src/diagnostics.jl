"""
    pdcond(A; kwargs...)

Compute the condition number `cond(A, 2)` for a positive definite matrix `A`.

This implementation uses `KrylovKit.svdsolve` to approximate the largest and smallest
singular values of `A` using only matrix-vector products.

`kwargs` are forwarded to `KrylovKit.svdsolve`.
"""
function pdcond(A; kwargs...)
    smin, smax = svdvals_extreme(A; kwargs...)
    return smax / smin
end

function svdvals_extreme(A; kwargs...)
    vals_max, _, _, info1 = KrylovKit.svdsolve(A, 1, :LR; kwargs...)
    if info1.converged == 0
        return _lastfirst(LinearAlgebra.svdvals(A))
    end
    # maybe we got all svdvals for the price of 1
    info1.converged == size(A, 1) && return _lastfirst(vals_max)
    vals_min, _, _, info2 = KrylovKit.svdsolve(A, 1, :SR; kwargs...)
    if info2.converged == 0
        return _lastfirst(LinearAlgebra.svdvals(A))
    end
    return first(vals_min), first(vals_max)
end

_lastfirst(x) = (last(x), first(x))

function ess_rhat(x)
    ess, rhat_bulk = MCMCDiagnosticTools.ess_rhat_bulk(x; maxlag=typemax(Int))
    rhat_tail = MCMCDiagnosticTools.rhat_tail(x)
    rhat = max.(rhat_bulk, rhat_tail)
    return (; ess, rhat)
end
