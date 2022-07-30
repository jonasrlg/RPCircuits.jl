# Parameter learning by batch Gradient Ascent

"""
Learn weights using the Gradient Ascent algorithm.
"""
mutable struct Gradient <: ParameterLearner
  circ::CCircuit
  score::Float64     # score (loglikelihood)
  prevscore::Float64 # for checking convergence
  tolerance::Float64 # tolerance for convergence criterion
  steps::Integer   # number of learning steps (epochs)
  minimumvariance::Float64 # minimum variance for Gaussian leaves
  # 1×N matrix filled with NaNs, where 'N' is the number of nodes
  marg_input::AbstractMatrix{<:Real}
  function Gradient(r::Node; gauss::Bool = false)
    circuit = compile(CCircuit, r; gauss)
    numnodes = length(circuit.C)
    marg_input = reshape(fill(NaN,numnodes), 1, numnodes)
    return new(circuit, NaN, NaN, 1e-4, 0, 0.5, marg_input)
  end
end
export Gradient

"""
Verifies if algorithm has converged.

Convergence is defined as absolute difference of score.
Requires at least 2 steps of optimization.
"""
converged(learner::Gradient) =
  learner.steps < 2 ? false : abs(learner.score - learner.prevscore) < learner.tolerance
export converged

# TODO: take filename os CSV File object as input and iterate over file to decrease memory footprint
"""
Improvement step for learning weights using the Gradient Ascent algorithm. Returns improvement in negative loglikelihood.

circ[i].weights[j] = circ[i].weights[k] * backpropagate(circ)[i]/sum(circ[i].weights[k] * backpropagate(circ)[i] for k=1:length(circ[i].children))

## Arguments

  - `learner`: Gradient ParamLearner struct
  - `data`: Data Matrix
  - `learningrate`: learning inertia, used to avoid forgetting in minibatch learning [default: 1.0 correspond to no inertia, i.e., previous weights are discarded after update]
  - `smoothing`: weight smoothing factor (= pseudo expected count) [default: 1e-4]
  - `learngaussians`: whether to also update the parameters of Gaussian leaves [default: false]
  - `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(
  learner::Gradient,
  Data::AbstractMatrix;
  learningrate::Float64 = 1e-4,
  smoothing::Float64 = 1e-4,
  learngaussians::Bool = false,
  minimumvariance::Float64 = learner.minimumvariance,
  verbose::Bool = false, 
  validation::AbstractMatrix = Data, 
  history = nothing,
  counter::Int64 = 100,
)

  numrows, numcols = size(Data)

  curr = learner.circ.C
  prev = learner.circ.P
  score = 0.0
  sumnodes = learner.circ.S
  if learngaussians
    gaussiannodes = learner.circ.gauss.G
  end

  numnodes = length(curr)

  V = Matrix{Float64}(undef, numrows, numnodes) # V[i][j] = log Cj(Di)
  Δ = Matrix{Float64}(undef, numrows, numnodes) # Δ[i][j] = log ΔC(Di)/ΔCj

  norm_V = Matrix{Float64}(undef, 1, numnodes) # norm_V[j] = log Cj(1)
  norm_Δ = Matrix{Float64}(undef, 1, numnodes) # norm_Δ[j] = log ΔC(1)/ΔCj 

  # Compute backward pass (values), considering normalization
  # LL = ∑d log(C(d)/C(1)) = ∑d [log C(d)] - |D|log C(1)
  LL = mplogpdf!(V, curr, Data)
  # Updated norm_V with values log Ci(1) and assigns log_norm_const
  # to the log of normalizing constant of the circuit, log C(1).
  log_norm_const = mplogpdf!(norm_V, curr, learner.marg_input)[1]

  # Compute forward pass (derivatives)
  pbackpropagate_tree!(Δ, curr, V; indicators = learner.circ.I)
  # Compute forward pass of marginalized variables
  pbackpropagate_tree!(norm_Δ, curr, norm_V; indicators = learner.circ.I)

  log_n = log(numrows)
  # Update sum weights
  Threads.@threads for i ∈ 1:length(sumnodes)
    s = sumnodes[i]
    S = curr[s]
    Δw = Vector{Float64}(undef, length(S.children))
    # Auxiliary variables
    logΔC = view(Δ, :, s) .- LL
    logΔC_norm = norm_Δ[s] - log_norm_const
    for (j, c) ∈ enumerate(S.children)
      # Let C be a circuit, then the derivative of log C 
      # w.r.t. the weight wⱼ (between nodes s and j) is
      # Δlog(C)/Δ wⱼ = (1/C) × (ΔC/Δwⱼ).
      # Using ΔC/Δwⱼ = (ΔC/ΔCₛ) × (ΔCₛ/Δwⱼ) = (ΔC/ΔCₛ) × Cⱼ,
      # we have that 
      # Δlog(C)/Δwⱼ = (ΔC/ΔCₛ) × Cⱼ ÷ C 
      # Thus, the log of Δlog(C), w.r.t. wⱼ, is
      # log[Δlog(C)] = log(ΔC/ΔCₛ) + log(Cⱼ) - log(C)
      #              = view(Δ, :, s) .+ view(V, :, c) .- LL
      #              = logΔC .+ view(V, :, c)

      # Similarly, the log of Δlog C(1) is
      # log{Δlog[C(1)]} = log[ΔC(1)/ΔCₛ] + log[Cⱼ(1)] - log[C(1)]
      #                 = norm_Δ[s]  + norm_V[c] - log_norm_const
      #                 = logΔC_norm + norm_V[c]

      # Therefore, we have that derivative of Δwⱼ is
      # ΔLL/Δwⱼ = ∑ᵢ [exp(log[Δlog(C)]ᵢ) - exp(log{Δlog[C(1)]})]/numrows
      #     = [∑ᵢ exp(log[Δlog(C)]ᵢ)]/numrows - exp(log{Δlog[C(1)]})
      #     = exp{log[∑ᵢ exp(log[Δlog(C)]ᵢ)]}/numrows - exp(log{Δlog[C(1)]})

      # Now, let lse ≔ logsumexp(log[Δlog(C)]). Then
      # ΔLL/Δwⱼ = exp(lse - log_n) - exp(log{Δlog[C(1)]})
      a = logsumexp(logΔC .+ view(V, :, c)) - log_n
      b = logΔC_norm + norm_V[c]

      # Trick to avoid NaNs results
      if a > b
        Δw[j] = exp(a+log(1-exp(b-a)))
      else
        Δw[j] = -exp(b+log(1-exp(a-b)))
      end
    end
    # Updated weights
    wt = S.weights .+ learningrate.*Δw
    S.weights[:] .= max.(wt, 1e-4) # Certifying that we don't get negative weights.
  end

  # Update Gaussian parameters
  if learngaussians
    Threads.@threads for i ∈ 1:length(gaussiannodes)
      g = gaussiannodes[i]
      G = curr[g]
      # Gaussian scope
      X = view(Data, :, G.scope)
      # Gaussian parameters
      μ, σ = G.mean, G.variance
      # Auxiliary variables
      aux = exp.(view(Δ, :, g) .+ view(V, :, g) .- LL)

      # Similarly to the previous calculation, we have
      # Δlog(C)/Δμ =  (ΔC/ΔG) × ΔG/Δμ ÷ C, and 
      # the derivative of C(1) is given by
      # Δlog[C(1)]/Δμ =  (ΔC(1)/ΔG) × ΔG(1)/Δμ ÷ C(1)

      # Since ΔG/Δμ = (x-μ)/σ G(x), we have that
      # ∑ Δlog(C)/Δμ = ∑ exp{log[ΔC/ΔG] - log(C)} × ΔG/Δμ
      #              = ∑ exp{log[ΔC/ΔG] - log(C)} × (x-μ)/σ × G(x)
      #              = ∑ exp{log[ΔC/ΔG] + log(G) - log(C)} × (x-μ)/σ
      #              = ∑ exp{view(Δ, :, g) .+ view(V, :, g) .- LL} × (x-μ)/σ
      #              = sum(aux .* (X.-μ)./σ)
      # Note that, this time, we cannot use the logsumexp trick.

      # Similarly, we have
      # Δlog[C(1)]/Δμ = exp{log[ΔC(1)/ΔG] - log[C(1)]} × ΔG/Δμ
      #               = 0 # Because G(1) is constant equal to 1

      # Hence, the derivative of the LL w.r.t. μ is
      # ΔLL/Δμ = 1/N ∑ [Δlog(C)/Δμ - Δlog[C(1)]/Δμ]
      #        = sum(aux .* (X.-μ)./σ)./N
      Δμ = sum(aux .* (X .- μ)./σ)./numrows
      G.mean += learningrate*Δμ
      
      # Analogously for the variance, we only have to change
      # the derivative of G w.r.t the variance σ
      # ΔG/Δσ = [(x-μ)^2/(2z^2) - 1/(2*σ)] G(x)
      # Therefore,
      # ΔLL/Δσ = 1/N ∑ [Δlog(C)/Δσ - Δlog[C(1)]/Δσ]
      #        = sum(aux .* ((X.- μ).^2./(2*σ^2) .- 1/(2*σ)))./N
      Δσ = sum(aux .* (((X .- μ).^2)./(2*σ^2) .- 1/(2*σ)))./numrows
      G.variance += learningrate*Δσ
      !(G.variance > minimumvariance) && (G.variance = minimumvariance)
    end
  end

  # TODO: Implementar callback
  #=
  if learner.steps % counter == 0
    m = size(validation, 1)
    # Computes Training LL before using the matrix V for validation
    if verbose 
      trainLL = sum(LL)/numrows - log_norm_const 
    end
    # loglikelihood of validation set
    if m <= numrows
      ll = mLL!(view(V, 1:m, :), curr, validation) - log_norm_const
    else
      ll = mLL!(Matrix{Float64}(undef, m, numnodes), curr, validation) - log_norm_const
    end
    if !isnothing(history)
      push!(history, ll)
    end
    if verbose
     println("Training LL: ", trainLL, " | Iteration $(learner.steps). η: $(learningrate), LL: $ll")
    end
  end
  =#

  #swap!(learner.circ)
  learner.steps += 1
  learner.prevscore = learner.score
  learner.score = -sum(LL)/numrows + log_norm_const
  #println("NLL = ", learner.score)
  #println(" LLH = ", sum(LL)/numrows, " / Norm const = $log_norm_const")

  return learner.prevscore - learner.score, V, Δ
end
export update
