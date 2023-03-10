# The following code is included to support alternative initializations of the inverse
# Hessian. See https://github.com/JuliaNLSolvers/Optim.jl/issues/955

"""
    LBFGS(;
        m = 10,
        alphaguess = LineSearches.InitialStatic(),
        linesearch = LineSearches.HagerZhang(),
        P = nothing,
        precondprep = (P, x) -> nothing,
        manifold = Optim.Flat(),
        init_invH0 = P === nothing ? init_invH0_nocedal_wright! : nothing,
    )

A modification of [`Optim.LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/)
to add the keyword `init_invH0`.

`init_invH0` is a mutating function with the signature `init_invH0!(invH0diag, s, y)`, `s`
is the change in the position, `y` is the change in the gradient, and `invH0diag` is the
diagonal of the initial inverse Hessian estimate from the previous iteration, which is
updated in-place.

See also: `init_invH0_nocedal_wright!`, `init_invH0_gilbert!`
"""
LBFGS

# Use the initial scaling guess from
# Nocedal & Wright (2nd ed), Equation (7.20)
init_invH0_nocedal_wright!(α, s, y) = fill!(α, dot(y, s) / sum(abs2, y))

# eq 4.9
# Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
# Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
function init_invH0_gilbert!(α, s, y)
    a = dot(y, Diagonal(α), y)
    b = dot(y, s)
    # TODO: work out what we should do when any elements of α are 0
    invα = (α .= inv.(α))  # avoid allocating a vector for the inverse
    c = dot(s, Diagonal(invα), s)
    return @. α = b / (a * invα + y^2 - (a / c) * (s * invα)^2)
end

#! format: off

# code adapted from https://github.com/JuliaNLSolvers/Optim.jl/blob/v1.7.4/src/multivariate/solvers/first_order/l_bfgs.jl
# Optim.jl is licensed under the MIT License:
# Copyright (c) 2012: John Myles White, Tim Holy, and other contributors. Copyright (c) 2016: Patrick Kofod Mogensen, John Myles White, Tim Holy, and other contributors. Copyright (c) 2017: Patrick Kofod Mogensen, Asbjørn Nilsen Riseth, John Myles White, Tim Holy, and other contributors.

# q is also a cache
function twoloop!(s,
                  gr,
                  rho,
                  dx_history,
                  dg_history,
                  m::Integer,
                  pseudo_iteration::Integer,
                  alpha,
                  q,
                  invH0diag,
                  init_invH0!,
                  precon)
    # Count number of parameters
    n = length(s)

    # Determine lower and upper bounds for loops
    lower = pseudo_iteration - m
    upper = pseudo_iteration - 1

    # Copy gr into q for backward pass
    copyto!(q, gr)
    # Backward pass
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i   = mod1(index, m)
        dgi = dg_history[i]
        dxi = dx_history[i]
        @inbounds alpha[i] = rho[i] * real(dot(dxi, q))
        @inbounds q .-= alpha[i] .* dgi
    end

    # Copy q into s for forward pass
    if precon === nothing && pseudo_iteration > 1
        i = mod1(upper, m)
        dxi = dx_history[i]
        dgi = dg_history[i]
        init_invH0!(invH0diag, dxi, dgi)
        s .= invH0diag .* q
    else
        # apply preconditioner if init_invH0! is nothing as the true setting
        # is essentially its own kind of preconditioning
        # (Note: preconditioner update was done outside of this function)
        Optim.ldiv!(s, precon, q)
    end
    # Forward pass
    for index in lower:1:upper
        if index < 1
            continue
        end
        i = mod1(index, m)
        dgi = dg_history[i]
        dxi = dx_history[i]
        @inbounds beta = rho[i] * real(dot(dgi, s))
        @inbounds s .+= dxi .* (alpha[i] - beta)
    end

    # Negate search direction
    rmul!(s, eltype(s)(-1))

    return
end

struct LBFGS{T, IL, L, Tprep, IH} <: Optim.FirstOrderOptimizer
    m::Int
    alphaguess!::IL
    linesearch!::L
    P::T
    precondprep!::Tprep
    manifold::Optim.Manifold
    init_invH0::IH
end

function LBFGS(; m::Integer = 10,
                 alphaguess = LineSearches.InitialStatic(), # TODO: benchmark defaults
                 linesearch = LineSearches.HagerZhang(),  # TODO: benchmark defaults
                 P=nothing,
                 precondprep = (P, x) -> nothing,
                 manifold::Optim.Manifold=Optim.Flat(),
                 init_invH0 = P === nothing ? init_invH0_nocedal_wright! : nothing)
    LBFGS(Int(m), _alphaguess(alphaguess), linesearch, P, precondprep, manifold, init_invH0)
end

Base.summary(::LBFGS) = "L-BFGS"

mutable struct LBFGSState{Tx, Tdx, Tdg, T, G} <: Optim.AbstractOptimizerState
    x::Tx
    x_previous::Tx
    g_previous::G
    rho::Vector{T}
    dx_history::Tdx
    dg_history::Tdg
    dx::Tx
    dg::Tx
    u::Tx
    f_x_previous::T
    twoloop_q
    twoloop_alpha
    pseudo_iteration::Int
    s::Tx
    invH0diag::Tx
    Optim.@add_linesearch_fields()
end
function reset!(method, state::LBFGSState, obj, x)
    Optim.retract!(method.manifold, x)
    Optim.value_gradient!(obj, x)
    Optim.project_tangent!(method.manifold, Optim.gradient(obj), x)

    state.pseudo_iteration = 0
end
function Optim.initial_state(method::LBFGS, options, d, initial_x)
    T = real(eltype(initial_x))
    n = length(initial_x)
    initial_x = copy(initial_x)
    Optim.retract!(method.manifold, initial_x)

    Optim.value_gradient!!(d, initial_x)

    Optim.project_tangent!(method.manifold, Optim.gradient(d), initial_x)
    LBFGSState(initial_x, # Maintain current state in state.x
              copy(initial_x), # Maintain previous state in state.x_previous
              copy(Optim.gradient(d)), # Store previous gradient in state.g_previous
              fill(T(NaN), method.m), # state.rho
              [similar(initial_x) for i = 1:method.m], # Store changes in position in state.dx_history
              [eltype(Optim.gradient(d))(NaN).*Optim.gradient(d) for i = 1:method.m], # Store changes in position in state.dg_history
              T(NaN)*initial_x, # Buffer for new entry in state.dx_history
              T(NaN)*initial_x, # Buffer for new entry in state.dg_history
              T(NaN)*initial_x, # Buffer stored in state.u
              real(T)(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), #Buffer for use by twoloop
              Vector{T}(undef, method.m), #Buffer for use by twoloop
              0,
              eltype(Optim.gradient(d))(NaN).*Optim.gradient(d), # Store current search direction in state.s
              fill!(similar(Optim.gradient(d)), 1),  # store initial inverse Hessian approximation in state.H0diag
              Optim.@initial_linesearch()...)
end

function Optim.update_state!(d, state::LBFGSState, method::LBFGS)
    n = length(state.x)
    # Increment the number of steps we've had to perform
    state.pseudo_iteration += 1

    Optim.project_tangent!(method.manifold, Optim.gradient(d), state.x)

    # update the preconditioner
    method.precondprep!(method.P, state.x)

    # Determine the L-BFGS search direction # FIXME just pass state and method?
    twoloop!(state.s, Optim.gradient(d), state.rho, state.dx_history, state.dg_history,
             method.m, state.pseudo_iteration,
             state.twoloop_alpha, state.twoloop_q, state.invH0diag, method.init_invH0, method.P)
    Optim.project_tangent!(method.manifold, state.s, state.x)

    # Save g value to prepare for update_g! call
    copyto!(state.g_previous, Optim.gradient(d))

    # Determine the distance of movement along the search line
    lssuccess = Optim.perform_linesearch!(state, method, Optim.ManifoldObjective(method.manifold, d))

    # Update current position
    state.dx .= state.alpha .* state.s
    state.x .= state.x .+ state.dx
    Optim.retract!(method.manifold, state.x)

    lssuccess == false # break on linesearch error
end


function Optim.update_h!(d, state, method::LBFGS)
    n = length(state.x)
    # Measure the change in the gradient
    state.dg .= Optim.gradient(d) .- state.g_previous

    # Update the L-BFGS history of positions and gradients
    rho_iteration = one(eltype(state.dx)) / real(dot(state.dx, state.dg))
    if isinf(rho_iteration)
        # TODO: Introduce a formal error? There was a warning here previously
        state.pseudo_iteration=0
        return true
    end
    idx = mod1(state.pseudo_iteration, method.m)
    state.dx_history[idx] .= state.dx
    state.dg_history[idx] .= state.dg
    state.rho[idx] = rho_iteration
    false
end

function Optim.trace!(tr, d, state, iteration, method::LBFGS, options, curr_time=time())
    Optim.common_trace!(tr, d, state, iteration, method, options, curr_time)
end

_alphaguess(a) = a
_alphaguess(a::Number) = LineSearches.InitialStatic(alpha=a)

#! format: on
