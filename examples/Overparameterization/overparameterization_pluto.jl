### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ b22f2888-eb89-11ec-1a91-037e834f7850
begin
	using Pkg
	Pkg.activate("/home/jonasrlg/code/RPCircuits.jl/")
	using RPCircuits, Random, DelimitedFiles, Plots
	using Distributions: rand, truncated, Normal
	using Statistics: mean, std
end

# ╔═╡ 952fb09c-c8e2-482a-b5f7-5c4731b454ff
function build_sum(dimension::Int; s::Float64 = 0.1)
    # Builds the sub-component of the variable w.r.t index = dimension
    a, na = Indicator(dimension, 1), Indicator(dimension, 0)
    # Initialize weights close to zero
    w = rand(truncated(Normal(0, s), 0, Inf), 2)
    return Sum([a,na], w)
end

# ╔═╡ 7f536850-8971-4459-8d83-4098fc160755
function build_component(ndim::Int; s::Float64 = 0.1)
    # Builds a component: a Prodcut node that has 'ndim' children,
    # where each children corresponds to a Sum node that takes
    # two Indicator nodes (of the same variable) as children
    sums = Vector{Node}(undef, ndim)
    for dim ∈ 1:ndim
        sums[dim] = build_sum(dim; s=s)
    end
    return Product(sums)
end

# ╔═╡ 08b14054-6f03-47c3-8468-f2301d408b9e
function build_pc(ncomponents::Int, depth::Int, ndim::Int; s::Float64 = 0.1)
    if depth == 1
        components = Vector{Node}(undef, ncomponents)
        for c ∈ 1:ncomponents
            components[c] = build_component(ndim; s=s)
        end
        # Initialize weights close to zero
        w = rand(truncated(Normal(0, s), 0, Inf), ncomponents)
        return Sum(components, w)
    end
	# Recursevily builds left and right circuits
	# Initialize weights close to zero
	w = rand(truncated(Normal(0, s), 0, Inf), 2)
	l = build_pc(ncomponents÷2,depth-1, ndim; s=s)
	r = build_pc(ncomponents÷2,depth-1, ndim; s=s)
	# Creates root
	return Sum([l,r], w)
end

# ╔═╡ 6b0cef57-ecf3-4b3d-842d-6b32ce9edb5f
begin
	# Dataset
	name = "nltcs"
	# Maxiumum depth of the circuit (Number of components = 2^k)
	k = 3
	ncomps = 2^k
	# Number of iterations
	maxiter = 500
	# Number of re-runs
	runs = 3
	# Learning rate
	η = 0.01
	# Variance of the gaussian that Initializes the random weights
	s = 0.1
end

# ╔═╡ 283daa8d-4b8f-4fad-b149-dc47c2c393d1
# Load Dataset
data, _, _ = twenty_datasets(name,as_df=false)

# ╔═╡ 6f69b05d-57dc-4a7a-a2d7-a6148a49d421
n, dim = size(data)

# ╔═╡ 1f994778-6e3a-4d1e-863e-fcf90896125a
begin
	# Train llh values for PCs with deepth 1, 2, ..., k
	llh = Vector{Matrix{Float64}}(undef, k)
	llh_mean = Vector{Vector{Float64}}(undef, k)
	for i ∈ 1:k
	    llh[i] = Matrix{Float64}(undef, runs, maxiter+1)
	end
end

# ╔═╡ 09fea983-ca9e-4fa0-90e1-886473fce9e9
# Runing experiments
for run ∈ 1:runs
	# d is the depth of the PC
	for d ∈ 1:k
		# Creates PC with weights close to zero and depth = d
		C = build_pc(ncomps,d,dim; s=s)
		L_C = Gradient(C)
		llh[d][run, 1] = -NLL(C, data) - log_norm_const(C) # Initial model LL
		for iter ∈ 1:maxiter
			update(L_C, data; learningrate=η) # One iteration of Gradient Descent
			llh[d][run, iter+1] = -L_C.score
		end
	end
end

# ╔═╡ 9fee0983-e9fb-437e-b8ac-8c8b2c32e0c1
begin
	plot(title = "Overparameterization on " * name, xlabel = "Iteration", 
    ylabel = "Train LLH", legend=:topleft)
	# Estimate performance over all re-runs.
	for fig ∈ 1:k
	    llh_mean[fig] = vec(mean(llh[fig], dims=1))
	    plot!(llh_mean[fig], label="Depth = $fig", ribbon=vec(std(llh[fig], dims=1)))
	end
	
	# Save to disk.
	savefig(name * "_pluto_experiment.pdf")
end

# ╔═╡ 82ee825d-f791-476b-944f-96a000e1d1bb
open(name * "_pluto_all_experiment.txt", "w") do file
    write(file, string(llh))
end

# ╔═╡ 8adcc17d-1d68-4627-a39b-a073f96c19e4
open(name * "_pluto_mean_experiment.txt", "w") do file
    write(file, string(llh_mean))
end

# ╔═╡ Cell order:
# ╠═b22f2888-eb89-11ec-1a91-037e834f7850
# ╠═952fb09c-c8e2-482a-b5f7-5c4731b454ff
# ╠═7f536850-8971-4459-8d83-4098fc160755
# ╠═08b14054-6f03-47c3-8468-f2301d408b9e
# ╠═6b0cef57-ecf3-4b3d-842d-6b32ce9edb5f
# ╠═283daa8d-4b8f-4fad-b149-dc47c2c393d1
# ╠═6f69b05d-57dc-4a7a-a2d7-a6148a49d421
# ╠═1f994778-6e3a-4d1e-863e-fcf90896125a
# ╠═09fea983-ca9e-4fa0-90e1-886473fce9e9
# ╠═9fee0983-e9fb-437e-b8ac-8c8b2c32e0c1
# ╠═82ee825d-f791-476b-944f-96a000e1d1bb
# ╠═8adcc17d-1d68-4627-a39b-a073f96c19e4
