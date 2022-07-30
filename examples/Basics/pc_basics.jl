### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ d4b51d48-1de8-4a22-a794-cd0240f62912
begin
	import Pkg
	Pkg.activate("/home/jonasrlg/code/RPCircuits.jl/")
end

# ╔═╡ c3fe146e-85c0-4100-aa80-6847221bdab4
using RPCircuits, Random

# ╔═╡ 3471dd29-8f2a-478b-b2ff-20b4c537d6e7
using Distributions: rand, truncated, Normal

# ╔═╡ 787e2109-0f75-4db2-b855-3e4ac6844bb2
md"""
# Building a Simple PC with RPCircuits
"""

# ╔═╡ 6caa1b24-a9cb-4421-9491-2cedc85eeb1c
md"""
The easyest way to build a PCs with RPCircuits uses a bottom-up strategy, definin first the `leaf` nodes, then `product` nodes, and lastly `sum` nodes. 

As an example, we start building a simple Circuit with 8 indicator (leafs), 4 products and 1 sum node.

![basic](spn_basic.png)
"""

# ╔═╡ 67ec8389-b5ac-49b8-a250-ac76f8cdaa6e
md"""
First we have the foundation of the PC, the **leaf** nodes. In this case the variables **a**, **b** and their respective negations **na** and **nb**. To declare an indicator leaf node using RPCircuits, we call the function `Indicator()`, that needs **two** arguments. The first argument simply refers to the variable's `scope` (the indices of the variables associated with the Indicator node). The second argument, corresponds to the `value` such that the Indicator node outputs `true` (if `value = 1.0`, the Indicator node outputs `true` only when its input is `1.0`). Using the packag we have
"""

# ╔═╡ 05345101-b38d-43bc-89bb-6923b28d0969
a, na, b, nb = Indicator(1, 1.0), Indicator(1, 0.), Indicator(2, 1.), Indicator(2, 0.)

# ╔═╡ 4fd88ddc-59c5-4edf-b3fd-fa7e03222926
md"""
Next, we have the second layer with 4 **product** nodes. To define a product `P`, we only need to call the function `Product(v)`, where `v`  is the vector containing all children of `P`. Hence, we can build the four products of our PC by
"""

# ╔═╡ ecdbff16-667e-482b-8033-94c8ebf06c51
P1, P2, P3, P4 = Product([a,b]), Product([a,nb]), Product([na,b]), Product([na,nb])

# ╔═╡ 6fd19b60-a612-4939-bab3-2ae2e294899b
md"""
At last, we have the **sum** node. To define a sum  node `S`, we have to call the function `Sum(v, w)`, where `v` is the vector of children of `S`; and `w` is the vector of corresponding weights. This can easily be done by
"""

# ╔═╡ d723ea8a-330b-4482-9b5c-9dc7aebf8e42
S = Sum([P1, P2, P3, P4], [0.4, 0.3, 0.2, 0.1])

# ╔═╡ 05a67dc2-3d28-4094-9e82-a8b272e73c1a
md"""
Hence, we have the circuit
![basic](spn_example.png)
"""

# ╔═╡ ba6a1d67-bb7e-4e6b-9c31-7330ac881f22
md"""
To see the `scope` of a circuit rooted at a node `C`, we can type `scope(C)`. Therefore, the scope of our circuit is
"""

# ╔═╡ 0ff84481-2ba6-452e-bf3e-8bc6d9dd909f
scope(S)

# ╔═╡ ece008a5-155a-4b80-9b61-aa63f78c4db3
md"""
Using RPCircuits, it is possible to randomly sample complete configurations of the variables associated with a circuit `C`. We can do this using the function `rand(C)`, that creates a sample according to the probability defined by the PC.
"""

# ╔═╡ 9f5648c3-c20b-4c1a-9f2a-a0651326efb1
sample = rand(S)

# ╔═╡ c439452a-5c7e-48f6-a258-ee04ad06e6a2
md"""
Passing a positive integer `N` to `rand(C, N)`, creates `N` random samples.
"""

# ╔═╡ 25252ee9-0f24-417e-bb61-b722c75ac54e
begin
	#Random.seed!(42)
	N = 1_000
	D = rand(S, N)
end

# ╔═╡ 45f641a4-7783-4645-81b1-fc5a78c41034
md"""
With the function `NLL(S, D)`, we have the `Negative Log-Likelihood` of the PC `S` w.r.t the dataset `D`.
"""

# ╔═╡ af9ab0c5-ea97-491e-b439-e394854f3bb3
println("Original model NLL = ", NLL(S, D))

# ╔═╡ 443cf2f1-b410-4092-8bd3-a858815cb29c
md"""
Suppose that we have an initial circuit `Sem = Sum([P1, P2, P3, P4], [0.25, 0.25, 0.25, 0.25])` and we want to learn the function `S` (such that configurations `(a,b)`, `(a,nb)`, `(na,b)` and `(na, nb)` have respective probabilities `0.4`, `0.3`, `0.2` and `0.1`). Firstly, we can check the initial `NLL` of our model `Sem` in relation to the dataset `D`.
"""

# ╔═╡ 0a69f856-367b-45de-ae68-ebba58f78b1c
Sem = Sum([P1, P2, P3, P4], [0.25, 0.25, 0.25, 0.25])

# ╔═╡ 5717246b-c4e5-41b7-92b9-794b830c907b
println("Initial EM NLL = ", NLL(Sem, D))

# ╔═╡ 795d0b7c-1e3c-4fd0-a121-9cf9e5cb8e8a
md"""
Now, we can pass both our circuit `Sem` and the dataset `D` as an input to the `EM` algorithm (Expectation-Maximization algorithm, more to know about it [here][murphy]). To do this, we first define the learner `L = SEM(S)`. Then, we have `m` calls of the `update` function, for `m` iterations of the `EM` algorithm.

[murphy]: https://probml.github.io/pml-book/book1.html
"""

# ╔═╡ 4718a5cc-2275-4859-88c5-c12564c90565
Lem = SEM(Sem);

# ╔═╡ 06eb534e-7177-46cc-9e58-e2273fdff41e
for i = 1:100
	update(Lem, D)
end

# ╔═╡ 6ef6276b-c00e-423e-8ff7-a68022ee37a5
println("Final EM Weights = ", Sem.weights)

# ╔═╡ a73c0df7-c166-4017-a6dc-22a4eb832890
md"""
At last, we can apply the `NLL` function another time, to see the improvement obtained by the learning process.
"""

# ╔═╡ 7366fa8d-2000-490d-8e9c-841a49650e18
println("Final EM NLL = ", NLL(Sem, D))

# ╔═╡ f7b23e54-3430-4e22-9075-6b72af2e748e
md"""
Similarly, we can use the Gradient Descent algorithm to learn a circuit `Sgrad` w.r.t the dataset `D`. In this process, we have a sligthly different approach, because we initalize the sum-weights near zero (more to know about it in: https://arxiv.org/abs/1905.0819).
"""

# ╔═╡ 7f71f89e-e449-4bcd-9055-1ac119f9df41
begin
	# Incialize weigths close to zero
	w = rand(truncated(Normal(0, 0.1), 0, Inf), 4)
end

# ╔═╡ ed6d6281-4df6-4306-a306-3213c2272891
Sgrad = Sum([P1, P2, P3, P4], w)

# ╔═╡ dee62d14-e4e8-4911-a10b-e21fc80a8637
md"""
Since the circuit `Sgrad` is not normalized, we need to compute its **normalizing constant**. We do this by using the function `log_norm_const` that outputs the `log` of the normalizing constant.
"""

# ╔═╡ 2e7b5f12-e6af-4ce3-9f15-58393d0a1b63
# Normalizing Constant of the circuit
norm_const = log_norm_const(Sgrad)

# ╔═╡ 1b32607f-3db7-4d7c-a924-f8431acfb0d2
md"""
Now, it is possible to obtain the real `NLL` of `Sgrad` w.r.t `D` by adding `norm_const` to `NLL(Sgrad, D)`
"""

# ╔═╡ 4e6fb3c7-b661-4206-b557-ea7bcc1af315
println("Initial Gradient NLL = ", NLL(Sgrad, D) + norm_const)

# ╔═╡ e12b8a95-5e44-4b06-b8ca-b563d76009f6
# ';' hides output
Lgrad = Gradient(Sgrad);

# ╔═╡ 9a241465-65c7-46cf-8846-816a5d603b5b
md"""
Finally, we can apply the Gradient Descent algorithm to `Sgrad` w.r.t `D` and then see the improvement obtained by the learning process.
"""

# ╔═╡ 0318da17-0c15-405e-8cb2-a573d2a07fee
for i = 1:100
	update(Lgrad, D; learningrate=0.01)
end

# ╔═╡ 67ad38b5-b733-4f3f-a441-4287c1aed07d
md"""
Now, we normalize our circuit `Sgrad` in such a way that it computes the same function (but multiplied by a normalization constant `1/Sgrad(1)`).
"""

# ╔═╡ 0dd3935f-522d-42ed-a4e1-e71cc197a9e3
normalize_circuit!(Sgrad) # Normalizing of circuit weights

# ╔═╡ ae491506-11a3-4a80-8bae-5c40f7fe9707
println("Final Gradient Weights(normalized) = ", Sgrad.weights)

# ╔═╡ 0ed64c0e-1bb5-489c-9668-b902970c9dd7
println("Final Gradient NLL = ", NLL(Sgrad, D))

# ╔═╡ Cell order:
# ╟─787e2109-0f75-4db2-b855-3e4ac6844bb2
# ╠═d4b51d48-1de8-4a22-a794-cd0240f62912
# ╠═c3fe146e-85c0-4100-aa80-6847221bdab4
# ╠═3471dd29-8f2a-478b-b2ff-20b4c537d6e7
# ╟─6caa1b24-a9cb-4421-9491-2cedc85eeb1c
# ╟─67ec8389-b5ac-49b8-a250-ac76f8cdaa6e
# ╠═05345101-b38d-43bc-89bb-6923b28d0969
# ╟─4fd88ddc-59c5-4edf-b3fd-fa7e03222926
# ╠═ecdbff16-667e-482b-8033-94c8ebf06c51
# ╟─6fd19b60-a612-4939-bab3-2ae2e294899b
# ╠═d723ea8a-330b-4482-9b5c-9dc7aebf8e42
# ╟─05a67dc2-3d28-4094-9e82-a8b272e73c1a
# ╟─ba6a1d67-bb7e-4e6b-9c31-7330ac881f22
# ╠═0ff84481-2ba6-452e-bf3e-8bc6d9dd909f
# ╟─ece008a5-155a-4b80-9b61-aa63f78c4db3
# ╠═9f5648c3-c20b-4c1a-9f2a-a0651326efb1
# ╟─c439452a-5c7e-48f6-a258-ee04ad06e6a2
# ╠═25252ee9-0f24-417e-bb61-b722c75ac54e
# ╟─45f641a4-7783-4645-81b1-fc5a78c41034
# ╠═af9ab0c5-ea97-491e-b439-e394854f3bb3
# ╟─443cf2f1-b410-4092-8bd3-a858815cb29c
# ╠═0a69f856-367b-45de-ae68-ebba58f78b1c
# ╠═5717246b-c4e5-41b7-92b9-794b830c907b
# ╟─795d0b7c-1e3c-4fd0-a121-9cf9e5cb8e8a
# ╠═4718a5cc-2275-4859-88c5-c12564c90565
# ╠═06eb534e-7177-46cc-9e58-e2273fdff41e
# ╠═6ef6276b-c00e-423e-8ff7-a68022ee37a5
# ╟─a73c0df7-c166-4017-a6dc-22a4eb832890
# ╠═7366fa8d-2000-490d-8e9c-841a49650e18
# ╟─f7b23e54-3430-4e22-9075-6b72af2e748e
# ╠═7f71f89e-e449-4bcd-9055-1ac119f9df41
# ╠═ed6d6281-4df6-4306-a306-3213c2272891
# ╟─dee62d14-e4e8-4911-a10b-e21fc80a8637
# ╠═2e7b5f12-e6af-4ce3-9f15-58393d0a1b63
# ╟─1b32607f-3db7-4d7c-a924-f8431acfb0d2
# ╠═4e6fb3c7-b661-4206-b557-ea7bcc1af315
# ╠═e12b8a95-5e44-4b06-b8ca-b563d76009f6
# ╟─9a241465-65c7-46cf-8846-816a5d603b5b
# ╠═0318da17-0c15-405e-8cb2-a573d2a07fee
# ╟─67ad38b5-b733-4f3f-a441-4287c1aed07d
# ╠═0dd3935f-522d-42ed-a4e1-e71cc197a9e3
# ╠═ae491506-11a3-4a80-8bae-5c40f7fe9707
# ╠═0ed64c0e-1bb5-489c-9668-b902970c9dd7
