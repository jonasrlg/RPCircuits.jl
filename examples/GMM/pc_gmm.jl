### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 6f1686e6-a2d1-423f-94aa-bfbe471e2761
begin
	import Pkg
	Pkg.activate("/home/jonasrlg/code/RPCircuits.jl/")
end

# ╔═╡ 6122c2d2-1af8-4f80-86d1-31d24a998784
using RPCircuits, Random, Distributions

# ╔═╡ 4b38b746-3a42-4f18-a34c-d9830cd32068
md"""
# Learning Gaussian Mixture Models with RPCircuits
"""

# ╔═╡ 130075c5-98c0-48a8-b7ff-94fc5db810d8
begin
	μ₁, σ₁ = -2.0, 1.2
	μ₂, σ₂ = 0.0, 1.0
	μ₃, σ₃ = 3.0, 2.5
	w₁,w₂, w₃ = 0.2, 0.5, 0.3
end

# ╔═╡ 87d61923-b791-45c5-b2f5-cce89e3bfd27
md"""
First, we create a gaussian with `mean = 0.3` and `variance = 1.0` using the `Distributions` package.
"""

# ╔═╡ cc8ed239-7e12-4317-bc55-68082300a688
gmm = MixtureModel(Normal[Normal(μ₁, σ₁),Normal(μ₂, σ₂),Normal(μ₃, σ₃)], [w₁,w₂,w₃])

# ╔═╡ 32f6628a-5e75-4707-bd95-d022666967b2
md"""
Then, we generate a dataset `D` with `N` samples of the previous distribution.
"""

# ╔═╡ 069b8f69-1d60-4f93-9d39-a5918c481682
begin
	#Random.seed!(42)
	N = 10_000
	samples = rand(gmm, N)
	D = reshape(samples, length(samples), 1)
end

# ╔═╡ 7c713f7c-6716-4dce-a233-77146bc807c0
md"""
Using `RPCircuits`, we create a `Gaussian Node` `G` with the same `mean` and `variance` as the previous gaussian distribution. Then, we apply the `NLL` function to see the Negative Log-Likelihood of `G` w.r.t. `D`.
"""

# ╔═╡ 8808e5be-c8a5-4334-bb47-67498bcb236a
begin
	G1, G2, G3 = Gaussian(1, μ₁, σ₁), Gaussian(1, μ₂, σ₂), Gaussian(1, μ₃, σ₃)
	S = Sum([G1, G2, G3], [w₁, w₂, w₃])
	println("Original model NLL = ", NLL(S, D))
end

# ╔═╡ f34da1f4-a0cd-40fb-97ea-b2d098d98106
md"""
Now, we create an arbitraty `Gaussian Node`that has both `mean` and `variance` different from the distribution `gauss`. Then, we apply the `EM` algorithm to learn a better distribution.
"""

# ╔═╡ ccaaef3c-7556-4a1d-bfd3-6e5086875062
begin
	G1em, G2em, G3em = Gaussian(1, -1., 3.), Gaussian(1, 1., 3.0), Gaussian(1, 4., 3.)
	Sem = Sum([G1em, G2em, G3em], [1/3, 1/3, 1/3])
	println("EM initial NLL = ", NLL(Sem, D))

	Lem = SEM(Sem; gauss=true)
	for i = 1:100
	    update(Lem, D; learngaussians=true)
	end
	
	println("EM final NLL = ", NLL(Sem, D))
	
	println("Sem = $Sem")
	println("G1em = $G1em")
	println("G2em = $G2em")
	println("G3em = $G3em")
end

# ╔═╡ 4b40be7c-ebaa-4912-a18a-36d0b19b87ef
md"""
Similarly to the example above, we create a `Gaussian Node` with both `mean` and `variance` differente from the distribution `gauss`. However, we apply the `Gradient Descent` algorithm in the learning process.
"""

# ╔═╡ bb831f22-4219-46af-8f07-610d662ff1a1
begin
	G1grad, G2grad, G3grad = Gaussian(1, -1., 3.), Gaussian(1, 1., 3.0), Gaussian(1, 4., 3.)
	# Incialize weigths close to zero
	w = rand(truncated(Normal(0, 0.1), 0.1, Inf), 3)
	Sgrad = Sum([G1grad, G2grad, G3grad], w)
	println("Grad initial NLL = ", NLL(Sgrad, D) + log_norm_const(Sgrad))

	Lgrad = Gradient(Sgrad, gauss=true)
	for i = 1:200
	    update(Lgrad, D; learningrate=0.1, learngaussians=true)
	end
	
	RPCircuits.normalize_circuit!(Sgrad; gauss=true)
	println("Grad final NLL = ", NLL(Sgrad, D))
	
	println("Sgrad = $Sgrad")
	println("G1grad = $G1grad")
	println("G2grad = $G2grad")
	println("G3grad = $G3grad")
end

# ╔═╡ Cell order:
# ╟─4b38b746-3a42-4f18-a34c-d9830cd32068
# ╠═6f1686e6-a2d1-423f-94aa-bfbe471e2761
# ╠═6122c2d2-1af8-4f80-86d1-31d24a998784
# ╠═130075c5-98c0-48a8-b7ff-94fc5db810d8
# ╟─87d61923-b791-45c5-b2f5-cce89e3bfd27
# ╠═cc8ed239-7e12-4317-bc55-68082300a688
# ╟─32f6628a-5e75-4707-bd95-d022666967b2
# ╠═069b8f69-1d60-4f93-9d39-a5918c481682
# ╟─7c713f7c-6716-4dce-a233-77146bc807c0
# ╠═8808e5be-c8a5-4334-bb47-67498bcb236a
# ╟─f34da1f4-a0cd-40fb-97ea-b2d098d98106
# ╠═ccaaef3c-7556-4a1d-bfd3-6e5086875062
# ╟─4b40be7c-ebaa-4912-a18a-36d0b19b87ef
# ╠═bb831f22-4219-46af-8f07-610d662ff1a1
