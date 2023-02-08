# wrapper to count the number of function evaluations and gradient evaluations
mutable struct EvalCountingProblem{P}
    const prob::P
    num_fun_evals::Int
    num_grad_evals::Int
    num_hess_evals::Int
end
EvalCountingProblem(prob) = EvalCountingProblem(prob, 0, 0, 0)

function num_evaluations(prob::EvalCountingProblem)
    return [prob.num_fun_evals, prob.num_grad_evals, prob.num_hess_evals]
end

function LogDensityProblems.capabilities(::Type{<:EvalCountingProblem{P}}) where {P}
    return LogDensityProblems.capabilities(P)
end

function LogDensityProblems.dimension(prob::EvalCountingProblem)
    return LogDensityProblems.dimension(prob.prob)
end

function LogDensityProblems.logdensity(prob::EvalCountingProblem, x)
    prob.num_fun_evals += 1
    return LogDensityProblems.logdensity(prob.prob, x)
end

function LogDensityProblems.logdensity_and_gradient(prob::EvalCountingProblem, x)
    prob.num_grad_evals += 1
    return LogDensityProblems.logdensity_and_gradient(prob.prob, x)
end

function LogDensityProblems.logdensity_gradient_and_hessian(prob::EvalCountingProblem, x)
    prob.num_hess_evals += 1
    return LogDensityProblems.logdensity_gradient_and_hessian(prob.prob, x)
end
