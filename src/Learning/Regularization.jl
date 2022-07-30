" Computes the entropy of the leaf node n."
function entropy_leaf(n::Leaf)
  # Entropy = -∑ₓ p(x)⋅log(p(x))
  if isindicator(n)
    return 0
  elseif isgaussian(n)
    return (log(2*π*n.variance)+1)/2
  elseif iscategorical(n)
    return -sum(n.values .* log.(n.values))
  elseif isbernoulli(n)
    prob = n.p
    return -(prob*log(prob) + (1-prob)*log(1-prob))
  end
end

" For each node n, computes the entropy of the sub-PC rooted at node n,
  as if it were a deterministic circuit."
function circuit_entropy(C::Vector{Node})
  k = length(C)
  entropy = Vector{Float64}(undef, k)
  @inbounds for i ∈ 1:k # from leaves to root
    N = C[i]
    if isprod(N)
      @inbounds entropy[i] = sum(entropy[N.children])
    elseif issum(N)
      @inbounds entropy[i] = sum(N.weights .* (entropy[N.children] .- log.(N.weights)))
    else # is leaf
      @inbounds entropy[i] = entropy_leaf(N)
    end
  end
  return entropy
end

" Calculates the entropy of a deterministic circuit."
function entropy(r::Node; gauss::Bool = false)
  circuit = compile(CCircuit, r; gauss)
  return circuit_entropy(circuit.C)[end]
end
export entropy

" Returns data entropy: -∑ᵢ Prob(Dᵢ)*log[Prob(Dᵢ)]"
function circuit_data_entropy(LL::AbstractVector)
  return -sum(exp.(LL).*LL)
end

" Calculates the entropy of a circuit given a dataset."
function data_entropy(r::Node, D::AbstractMatrix; gauss::Bool = false)
  circuit = compile(CCircuit, r; gauss)
  C = circuit.C
  V = Matrix{Float64}(undef, size(D)[1], length(C))
  LL = mplogpdf!(V, C, D)
  return circuit_data_entropy(LL)
end
export data_entropy

" Computes the sum of the expected flows over all data samples.
  When applied to a deterministic PC, returns the deterministic flow."
function expected_flows(C::Vector{Node}, summap::AbstractDict{Int, Int}, V::AbstractMatrix)
  numrows, numnodes = size(V)
  numsum = length(summap)

  # Expected Flows
  EF_edges = Matrix{Vector{Float64}}(undef, numrows, numsum) # EF[x,n][c] = EF(n,c)(x)
  EF_nodes = Matrix{Float64}(undef, numrows, numnodes) # EF[x,n] = EF(n)(x)
  EF_nodes[:, end] .= 1 # Starts backwards pass
  
  Threads.@threads for x ∈ 1:numrows
    for i ∈ numnodes:-1:1 # from root to leaves
      N = C[i]
      # If we have a leaf node, there is nothing to computes
      if isleaf(N) continue
      # If N is a sum node, we compute de Expected Flow
      elseif issum(N)
        # Hiearquical index of the sum node
        s = summap[i]
        EF_edges[x,s] = Vector{Float64}(undef, length(N.children))
        d = EF_nodes[x, i] # EF(N)(x)
        v = V[x, i] # log Prₙ(x)
        for (j, c) ∈ enumerate(N.children)
          # EF(N,c) = EF(N) ⋅ Θ(N,c) ⋅ [Pr[c]/Pr[N]]
          EF_edges[x,s][j] = d * N.weights[j] * context(v, V[x, c])
          # Propagates the Expected_Flows for its children
          # EF(c) = ∑ₚ EF(p,c), where p ∈ pa(c)
          EF_nodes[x,c] += EF_edges[x,s][j]
        end
      # If N is a product node, we only propagate the expected flow
      elseif isprod(N)
        d = EF_nodes[x, i] # EF(N)(x)
        for c ∈ N.children
          # Propagates the Expected_Flows for its children
          # EF(c) = ∑ₚ EF(p,c), where p ∈ pa(c) and
          # EF(N,c) = EF(N)
          EF_nodes[x,c] += d
        end
      end
    end
  end

  # For each sum node edge, return the sum of the Expected Flows over all data samples.
  return sum(EF_edges, dims=1) # ∑ₓ EF(s,c)(x)
end

" Certificies that we are not dividing by zero."
function context(a::Float64, b::Float64)
  if b > -Inf
    return exp(a-b)
  end
  return 0
end

" Maps the indices of all sum nodes to an topological index. If a sum node
is a descendent of another sum node, the index of the former will be greater
than the latter."
function map_sum_index(C::Vector{Node}, sumnodes::Vector{UInt})::AbstractDict{Int, Int}
  summap = Dict{Int, Int}()
  for i ∈ 1:length(C)
    N = C[i]
    if issum(N)
      # Associates the Circuit index with the hiearquical index of the sum node
      summap[i] = findfirst(index -> index == i, sumnodes)
    end
  end
  return summap
end

" Entropy regularization based on the algorithm 'PC Entropy regularization', developed 
  by Liu and Van den Broeck (Tractable Regularization of Probabilistic Circuits)."
function entropy_regularization!(C::Vector{Node}, summap::AbstractDict{Int, Int}, V::AbstractMatrix; 
                                entropy_reg::Float64=1e-4)
  numrows, numnodes = size(V)
  
  # node_prob is definided recursively as follows
  # node_prob(root) ≔ 1
  # node_prob(n) = ∑ₚ node_prob(p) + ∑ₛ node_prob(s),
  # where p are the product nodes that are parents of n, 
  # and s are the sum nodes that are parents of n. 
  node_prob = Vector{Float64}(undef, numnodes)
  node_prob[end] = 1 # the root
    
  # The entropy (assuming determinism) of the sub-PC rooted at n
  entropy = circuit_entropy(C)
  
  # Mean of the Expected Flows, for each sum node edge
  EF = expected_flows(C, summap, V)./numrows

  for n ∈ numnodes:-1:1 # nodes in preorder (starting from root)
    N = C[n]
    # If N is a leaf, there is nothing to update
    if isleaf(N) continue

    # If N is a sum node, we have an update of the weights using
    # Newton's Method.
    elseif issum(N)
      # Hiearquical index of the sum node
      s = summap[n]
      # Parameters for solving the set the equations
      # that maximizes the entropy
      d = EF[s] # Expected Flows of all edges of N
      b = entropy_reg*node_prob[n]
      children_entropy = entropy[N.children]

      # Next, we maximize the entropy regularization objective.
      # This is done by solving (using Newton's method)  the 
      # following set of equations
      # (1) y = d/θᵢ - b⋅logθᵢ + b⋅entropy[cᵢ],
      # (2) ∑ θᵢ = 1, θᵢ ≥ 0,
      # where θᵢ is the weight between the sum node s and
      # the respective child node cᵢ

      log_weights = log.(N.weights)
      log_weights .-= logsumexp(log_weights) # Normalizing the weights
      for y_iter ∈ 1:3
        # Estimation of y by taking the mean of equation (1) over all children
        y = sum(d.*exp.(-log_weights) .+ b*(children_entropy .- log_weights))/length(N.children)
        for param_iter ∈ 1:4
          # Here, we update the log weights using Newton's method.
          # Since d⋅exp(-φᵢ) - b⋅φᵢ + b⋅entropy[cᵢ] = y, we can
          # apply Newton's Method to find the root of the following
          # function f(x) = d⋅exp(-x) - b⋅x + b⋅γᵢ - y.
          # As df/dx = -[d⋅exp(-x) + b], the update of φ will be as
          # follows
          # φᵢ = φᵢ - f(φᵢ)/[df(φᵢ)/dx] = φᵢ + f(φᵢ)/(d⋅exp(-x) + b)
          ratio = d .* exp.(-log_weights)
          log_weights .+= (ratio .+ b*(children_entropy .- log_weights) .- y)./(ratio .+ b)
          log_weights .-= logsumexp(log_weights) # Normalization
        end
      end

      # Updates sum node parameters
      N.weights .= exp.(log_weights)

      # Updates children node_prob as follows
      # node_prob[cᵢ] += exp(φᵢ) ⋅ node_prob[n]
      node_prob[N.children] .+= N.weights .* node_prob[n]

    # If N is a product node, we only propagate node_prob
    elseif isprod(N)
      # Updates children node_prob as follows
      # node_prob[cᵢ] += node_prob[n]
      node_prob[N.children] .+= node_prob[n] 
    end
  end
end