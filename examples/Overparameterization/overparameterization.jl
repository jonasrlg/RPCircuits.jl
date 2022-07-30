using Pkg
Pkg.activate("/home/jonasrlg/code/RPCircuits.jl/")
using RPCircuits, Random, DelimitedFiles, Plots
using Distributions: rand, truncated, Normal
using Statistics: mean, std

function build_sum(dimension::Int; rand_init::Bool = true, var::Float64 = 0.1)
    # Builds the sub-component of the variable w.r.t index = dimension
    a, na = Indicator(dimension, 1), Indicator(dimension, 0)
    if rand_init
        # Initialize weights close to zero
        w = rand(truncated(Normal(0, var), 0, Inf), 2)
    else
        w = [0.5,0.5]
    end
    return Sum([a,na], w)
end

function build_component(ndim::Int; rand_init::Bool = true, var::Float64 = 0.1)
    # Builds a component: a Prodcut node that has 'ndim' children,
    # where each children corresponds to a Sum node that takes
    # two Indicator nodes (of the same variable) as children
    sums = Vector{Node}(undef, ndim)
    for dim ∈ 1:ndim
        sums[dim] = build_sum(dim; rand_init=rand_init, var=var)
    end
    return Product(sums)
end

#=
function build_pc(ncomponents::Int, depth::Int, ndim::Int; rand_init::Bool = true, var::Float64 = 0.1)
    if ncomponents <= 0
        # If the initial input is invalid
        return
    end
    if depth == 1
        components = Vector{Node}(undef, ncomponents)
        for c ∈ 1:ncomponents
            components[c] = build_component(ndim; rand_init=rand_init, var=var)
        end
        if rand_init
            # Initialize weights close to zero
            w = rand(truncated(Normal(0, var), 0, Inf), ncomponents)
        else
            w = ones(ncomponents)./ncomponents
        end
    end
    mid = ncomponents ÷ 2
    if mid > 0
        # Recursevily builds left and right circuits
        l = build_pc(ncomponents - mid, depth-1, ndim; rand_init=rand_init, var=var)
        r = build_pc(mid, depth-1, ndim; rand_init=rand_init, var=var)
        if rand_init
            # Initialize weights close to zero
            w = rand(truncated(Normal(0, var), 0, Inf), 2)
        else
            w = [0.5,0.5]
        end
        # Creates root
        return Sum([l,r], w)
    else
        # If 'mid' is not greater than 0, ncomponents is equal to 1.
        # Thus, we create a single component 's'.
        if rand_init
            w = rand(truncated(Normal(0, s), 0, Inf), 1)
        else
            w = [1]
        s = build_pc(1, depth-1, ndim; rand_init=rand_init, var=var)
        return Sum([s],w)
    end
end
=#

function build_pc(ncomponents::Int, depth::Int, ndim::Int; rand_init::Bool = true, var::Float64 = 0.1)
    if depth == 1
        components = Vector{Node}(undef, ncomponents)
        for c ∈ 1:ncomponents
            components[c] = build_component(ndim; rand_init=rand_init, var=var)
        end
        if rand_init
            # Initialize weights close to zero
            w = rand(truncated(Normal(0, var), 0, Inf), ncomponents)
        else
            w = ones(ncomponents)./ncomponents
        end
        return Sum(components, w)
    end
	# Recursevily builds left and right circuits
    l = build_pc(ncomponents÷2, depth-1, ndim; rand_init=rand_init, var=var)
	r = build_pc(ncomponents÷2, depth-1, ndim; rand_init=rand_init, var=var)
    if rand_init
        # Initialize weights close to zero
	    w = rand(truncated(Normal(0, var), 0, Inf), 2)
	else
        w = [0.5,0.5]
    end
    # Creates root
	return Sum([l,r], w)
end

######################
# Default variables. #
######################

# Dataset
name = "nltcs"
# Maxiumum depth of the circuit (Number of components = 2^k)
k = 3
ncomps = 2^k
# Number of iterations
maxiter = 500
# Number of re-runs
runs = 5
# Learning rate
η = 0.01
# Variance of the gaussian that Initializes the random weights
init_var = 0.1

if !isempty(ARGS)
    name = ARGS[1]
    if length(ARGS) > 1
        k = parse(Int64, ARGS[2])
        ncomps = 2^k
        if length(ARGS) > 2
            maxiter = parse(Int64, ARGS[3])
            if length(ARGS) > 3
                runs = parse(Int64, ARGS[4])
                if length(ARGS) > 4
                    η = parse(Float64, ARGS[5])
                    if length(ARGS) > 5
                        init_var = parse(Float64, ARGS[6])
                    end
                end
            end
        end
    end
end

println("Dataset = $name / Depth = $k / MaxIter = $maxiter / Learning Rate = $η / Weight Variance = $init_var")

# Load Dataset
data, _, _ = twenty_datasets(name,as_df=false)
n, dim = size(data)

# Train llh values for PCs with deepth 1, 2, ..., k
llh_Grad = Vector{Matrix{Float64}}(undef, k)
llh_EM = Vector{Matrix{Float64}}(undef, k)
for i ∈ 1:k
    llh_Grad[i] = Matrix{Float64}(undef, runs, maxiter)
    llh_EM[i] = Matrix{Float64}(undef, runs, maxiter)
end

# Runing Gradient experiments
println("Gradient experiments:")
for run ∈ 1:runs
    println("   Run $run:")
    # d is the depth of the PC
    for d ∈ 1:k
        println("       Depth = $d")
        # Creates PC with weights close to zero and depth = d
        C = build_pc(ncomps, d, dim; rand_init=true, var=init_var)
        L_C = Gradient(C)
        for iter ∈ 1:maxiter
            update(L_C, data; learningrate=η) # One iteration of Gradient Descent
            llh_Grad[d][run, iter] = -L_C.score
        end
    end
end

# Runing EM experiments
println("EM experiments:")
for run ∈ 1:runs
    println("   Run $run:")
    # d is the depth of the PC
    for d ∈ 1:k
        println("      Depth = $d")
        # Creates PC with weights close to zero and depth = d
        C = build_pc(ncomps, d, dim; rand_init=true, var=init_var)
        # Normalizes PC weights to apply EM algorithm
        normalize_circuit!(C)
        L_C = SEM(C)
        for iter ∈ 1:maxiter
            update(L_C, data) # One iteration of EM with default learning rate
            llh_EM[d][run, iter] = -L_C.score
        end
    end
end

# Control experiment
# Creating mixture model with uniform weights and applying EM
llh_control = Vector{Float64}(undef, maxiter)
println("Control experiment")
# Creates PC with weights close to zero and depth = d
C = build_pc(ncomps, 1, dim; rand_init=false)
L_C = SEM(C)
for iter ∈ 1:maxiter
    update(L_C, data) # One iteration of EM with default learning rate
    llh_control[iter] = -L_C.score
end

#########################
# Visualize Experiment. #
#########################

llh_Grad_mean = Vector{Vector{Float64}}(undef, k)
llh_EM_mean = Vector{Vector{Float64}}(undef, k)
# Estimate performance over all re-runs.
for fig ∈ 1:k
    llh_Grad_mean[fig] = vec(maximum(llh_Grad[fig], dims=1))
    llh_EM_mean[fig] = vec(maximum(llh_EM[fig], dims=1))
end

println("Saving data")
#writedlm(name*"_grad.csv",  llh_Grad, ',')
writedlm(name*"_grad_mean.csv",  llh_Grad_mean, ',')
#writedlm(name*"_em.csv",  llh_EM, ',')
writedlm(name*"_em_mean.csv",  llh_EM_mean, ',')
writedlm(name*"_control.csv",  llh_control, ',')


println("Creating plots")

function experiment_plots(name::String;
    sufix::String = "",
    grad_mean::Union{Vector{Vector{Float64}}, Nothing} = nothing, 
    em_mean::Union{Vector{Vector{Float64}}, Nothing} = nothing, 
    control::Union{Vector{Float64}, Nothing} = nothing)
    plot(title = "Overparameterization on " * name, xlabel = "Iteration", 
        ylabel = "Train LLH", legend=:bottomright)
    if grad_mean ≠ nothing
        for fig ∈ 1:length(grad_mean)
            plot!(grad_mean[fig], label="Gradient - Depth = $fig")
        end
    end
    if em_mean ≠ nothing
        for fig ∈ 1:length(em_mean)
            plot!(em_mean[fig], label="EM - Depth = $fig")
        end
    end
    if control ≠ nothing
        plot!(llh_control, label="EM - Control")
    end
    # Save to disk.
    savefig(name*sufix*".pdf")
end

function batch_plots(name::String;
    sufix::String = "", 
    grad_mean::Union{Vector{Vector{Float64}}, Nothing} = nothing, 
    em_mean::Union{Vector{Vector{Float64}}, Nothing} = nothing, 
    control::Union{Vector{Float64}, Nothing} = nothing)
    # Plot with all experiments
    experiment_plots(name; sufix="_all"*sufix, grad_mean=grad_mean, em_mean=em_mean, control=control)
    # Ploting only Gradient and control
    experiment_plots(name; sufix="_grad"*sufix, grad_mean=grad_mean, control=control)
    # Ploting only EM and control
    experiment_plots(name; sufix="_em"*sufix, em_mean=em_mean, control=control)
end

batch_plots(name; grad_mean=llh_Grad_mean, em_mean=llh_EM_mean, control=llh_control)

if maxiter >= 750
    for i ∈ 1:k
        llh_Grad_mean[i] = llh_Grad_mean[i][1:500]
        llh_EM_mean[i] = llh_EM_mean[i][1:500]
    end
    llh_control = llh_control[1:500]
    batch_plots(name; sufix="_500", grad_mean=llh_Grad_mean, em_mean=llh_EM_mean, control=llh_control)
end

if maxiter >= 300
    for i ∈ 1:k
        llh_Grad_mean[i] = llh_Grad_mean[i][1:200]
        llh_EM_mean[i] = llh_EM_mean[i][1:200]
    end
    llh_control = llh_control[1:200]
    batch_plots(name; sufix="_200", grad_mean=llh_Grad_mean, em_mean=llh_EM_mean, control=llh_control)
end

if maxiter >= 150
    for i ∈ 1:k
        llh_Grad_mean[i] = llh_Grad_mean[i][1:100]
        llh_EM_mean[i] = llh_EM_mean[i][1:100]
    end
    llh_control = llh_control[1:100]
    batch_plots(name; sufix="_100", grad_mean=llh_Grad_mean, em_mean=llh_EM_mean, control=llh_control)
end

if maxiter >= 75
    for i ∈ 1:k
        llh_Grad_mean[i] = llh_Grad_mean[i][1:50]
        llh_EM_mean[i] = llh_EM_mean[i][1:50]
    end
    llh_control = llh_control[1:50]
    batch_plots(name; sufix="_50", grad_mean=llh_Grad_mean, em_mean=llh_EM_mean, control=llh_control)
end

if maxiter >= 50
    for i ∈ 1:k
        llh_Grad_mean[i] = llh_Grad_mean[i][1:25]
        llh_EM_mean[i] = llh_EM_mean[i][1:25]
    end
    llh_control = llh_control[1:25]
    batch_plots(name; sufix="_25", grad_mean=llh_Grad_mean, em_mean=llh_EM_mean, control=llh_control)
end