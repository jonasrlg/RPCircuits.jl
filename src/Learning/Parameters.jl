using Random

"""
    Parameter Learning Algorithm
"""
abstract type ParameterLearner end

"""
    backpropagate!(diff::Dict{Node, Float64}, r::Node, values::Dict{Node, Float64})

Computes derivatives for values propagated in log domain and stores results in given vector diff.
"""
function backpropagate!(diff::Dict{Node, Float64}, C::Vector{Node}, values::Dict{Node, Float64})
  # Assumes network has been evaluted at some assignment using logpdf!
  # Backpropagate derivatives
  for i ∈ length(C):-1:1
    n = C[i]
    if issum(n)
      for (j, c) in enumerate(n.children)
        !haskey(diff, c) && (diff[c] = 0)
        diff[c] += n.weights[j] * diff[n]
      end
    elseif isprod(n)
      for c in n.children
        !haskey(diff, c) && (diff[c] = 0)
        if isfinite(values[c])
          # @assert isfinite(exp(values[i]-values[j]))  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(exp(values[i]-values[j]))"
          diff[c] += diff[n] * exp(values[n] - values[c])
        else
          δ = exp(sum(values[k] for k in n.children if k ≠ c))
          # @assert isfinite(δ)  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(δ)"
          diff[c] += diff[n] * δ
        end
      end
    end
  end
  return nothing
end
function backpropagate!(diff::Vector{Float64}, circ::Vector{Node}, values::Vector{Float64})
  fill!(diff, 0.0)
  @inbounds diff[end] = 1.0
  for i in length(circ):-1:1
    @inbounds node = circ[i]
    if issum(node)
      @inbounds d = diff[i]
      @inbounds for (j, c) in enumerate(node.children)
        diff[c] += node.weights[j] * d
      end
    elseif isprod(node)
      @inbounds d = diff[i]
      @inbounds for j in node.children
        if isfinite(values[j])
          # @assert isfinite(exp(values[i]-values[j]))  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(exp(values[i]-values[j]))"
          diff[j] += d * exp(values[i] - values[j])
        else
          δ = exp(sum(values[u] for u in node.children if u ≠ j))
          # @assert isfinite(δ)  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(δ)"
          diff[j] += d * δ
        end
      end
    end
  end
  return nothing
end

function pbackpropagate_tree!(Δ::AbstractMatrix{<:Real}, C::Vector{Node}, V::AbstractMatrix{<:Real};
  indicators::Union{AbstractVector{<:Integer}, Nothing} = nothing)
  # lim = -floatmax(eltype(Δ))
  if isnothing(indicators)
    Δ .= 0.0
  else
    Threads.@threads for i ∈ 1:length(indicators)
      Δ[:,indicators[i]] .= 0.0
    end
    Δ[:,end] .= 0.0
  end
  n, m = size(Δ)
  contains_indicators = false
  I = prepare_indices(n)
  L = ReentrantLock()
  Threads.@threads for j ∈ 1:length(I)
    R = I[j]
    infs = BitVector(undef, length(R))
    # Columns: nodes
    for i ∈ m:-1:1 # from root to leaves
      N = C[i]
      if isleaf(N) continue end
      d = view(Δ, R, i)
      if issum(N)
        for (j, c) ∈ enumerate(N.children)
          if C[c] isa Indicator
            # Indicator nodes are computed differently, since they may repeat (and so the structure is not
            # tree-shaped). Compute in normal space so as to minimize numerical errors.
            contains_indicators = true
            lock(L); try Δ[R,c] .+= N.weights[j] * exp.(d) finally unlock(L) end
          else
            Δ[R,c] .= log(N.weights[j]) .+ d
          end
        end
      elseif isprod(N)
        v = view(V, R, i)
        for c ∈ N.children
          u = view(V, R, c)
          map!(isinf, infs, u)
          δ = Vector{Float64}(undef, length(R))
          noninfs = findall(!, infs)
          (length(noninfs) > 0) && (δ[noninfs] .= view(d, noninfs) .+ view(v, noninfs) .- view(u, noninfs))
          infs_idx = findall(infs)
          if length(infs_idx) > 0
            for j ∈ infs_idx
              δ[j] = d[j] + sum(V[R[j],x] for x ∈ N.children if x ≠ c)
            end
          end
          # setinvalid!(δ, lim)
          if C[c] isa Indicator
            contains_indicators = true
            lock(L); try Δ[R,c] .+= exp.(δ) finally unlock(L) end
          else
            Δ[R,c] .= δ
          end
        end
      end
    end
  end
  if contains_indicators
    for i ∈ 1:m
      (C[i] isa Indicator) && (Δ[:,i] .= log.(view(Δ, :, i)))
    end
  end
  return nothing
end

function backpropagate_tree!(Δ::AbstractMatrix{<:Real}, C::Vector{Node}, V::AbstractMatrix{<:Real})
  lim = -floatmax(eltype(Δ))
  Δ .= 0.0
  n, m = size(Δ)
  contains_indicators = false
  # Columns: nodes
  for i ∈ m:-1:1 # from root to leaves
    N = C[i]
    d = view(Δ, :, i)
    if isleaf(N) continue end
    if issum(N)
      for (j, c) ∈ enumerate(N.children)
        if C[c] isa Indicator
          # Indicator nodes are computed differently, since they may repeat (and so the structure is not
          # tree-shaped). Compute in normal space so as to minimize numerical errors.
          contains_indicators = true
          Δ[:,c] .+= N.weights[j] * exp.(d)
        else
          Δ[:,c] .= log(N.weights[j]) .+ d
        end
      end
    elseif isprod(N)
      v = view(V, :, i)
      for c ∈ N.children
        δ = d .+ v .- view(V, :, c)
        setinvalid!(δ, lim)
        if C[c] isa Indicator
          contains_indicators = true
          Δ[:,c] .*= exp.(δ)
        else
          Δ[:,c] .= δ
        end
      end
    end
  end
  if contains_indicators
    for i ∈ 1:m
      (C[i] isa Indicator) && (Δ[:,i] .= log.(view(Δ, :, i)))
    end
  end
  return nothing
end

function backpropagate_tree!(diff::AbstractVector{<:Real}, circ::Vector{Node}, layers::Vector{Vector{UInt}}, values::AbstractVector{Float64})
  fill!(diff, 0.0)
  @inbounds diff[end] = 1.0
  for l in 1:length(layers)
    Threads.@threads for i ∈ layers[l]
      @inbounds node = circ[i]
      if issum(node)
        @inbounds d = diff[i]
        @inbounds for (j, c) in enumerate(node.children)
          diff[c] += node.weights[j] * d
        end
      elseif isprod(node)
        @inbounds d = diff[i]
        @inbounds for j in node.children
          if isfinite(values[j])
            # @assert isfinite(exp(values[i]-values[j]))  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(exp(values[i]-values[j]))"
            diff[j] += d * exp(values[i] - values[j])
          else
            δ = exp(sum(values[u] for u in node.children if u ≠ j))
            # @assert isfinite(δ)  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(δ)"
            diff[j] += d * δ
          end
        end
      end
    end
  end
  return nothing
end

"""
    backpropagate(r::Node, values::Dict{Node, Float64})::Dict{Node, Float64}

Computes derivatives for given vector of values propagated in log domain.

Returns vector of derivatives.
"""
function backpropagate(r::Node, values::Dict{Node, Float64})::Dict{Node, Float64}
  diff = Dict{Node, Float64}()
  backpropagate!(diff, r, values)
  return diff
end

"""
Random initialization of weights
"""
function initialize(learner::ParameterLearner)
  r = learner.root
  function f(i::Int, n::Node)
    if issum(n)
      @inbounds Random.rand!(n.weights)
      @inbounds n.weights ./= sum(n.weights)
      @assert sum(n.weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(n.weights)) | $(n.weights)"
    elseif n isa Gaussian
      @inbounds n.mean = rand()
      @inbounds n.variance = 1.0
    end
  end
  foreach(f, r; rev = false)
end
export initialize
