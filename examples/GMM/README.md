# Learning Gaussian Nodes with RPCircuits

In this section, we show how you can train `Nodes` of the type `Gaussian` by using the `RPCircuits` package. See `pc_gauss.ipynb` for an interactive notebook.

First, we create a gaussian with `mean = 0.3` and `variance = 1.0` using the `Distributions` package.

```julia
julia> mean, var = 0.3, 1.0;

julia> gauss = Distributions.Normal(mean, var)
Normal{Float64}(μ=0.3, σ=1.0)
```

Then, we generate a dataset `D` with `N` samples of the previous distribution.

```julia
julia> N = 100_000;

julia> samples = Distributions.rand(gauss, N);

julia> D = reshape(samples, length(samples), 1)
100000×1 Matrix{Float64}:
  2.3564256040912346
 -0.7689958578147957
 -1.066765767584314
 -0.7007263948719711
  0.24121541475799688
 -0.6142212883260776
 -0.08726674329080225
  ⋮
  0.6112053815225873
  0.7130353648174477
  0.009808856838898483
  0.8129856057040985
 -0.5264521795372346
  0.3907007707071724
 -0.9603826206178665
```

Using `RPCircuits`, we create a `Gaussian Node` `G` with the same `mean` and `variance` as the previous gaussian distribution. Then, we apply the `NLL` function to see the Negative Log-Likelihood of `G` w.r.t. `D`.

```julia
julia> G = RPCircuits.Gaussian(1, mean, var)
Circuit with 1 node (0 sums, 0 products, 1 leaf) and 1 variable:
  1 : gaussian 1 0.3 1.0

julia> println("Original model NLL = ", NLL(G, D))
Original model NLL = 1.4213589713858874
```

Now, we create an arbitraty `Gaussian Node`that has both `mean` and `variance` different from the distribution `gauss`.

```julia
julia> G_em = RPCircuits.Gaussian(1, -0.15, 2.5)
Circuit with 1 node (0 sums, 0 products, 1 leaf) and 1 variable:
  1 : gaussian 1 -0.15 2.5

julia> L_em = SEM(G_em; gauss=true);

julia> println("EM initial NLL = ", NLL(G_em, D))
EM initial NLL = 1.6182111383613536

julia> for i = 1:50
           update(L_em, D; learngaussians=true, verbose=false)
       end

julia> println("EM final NLL = ", NLL(G_em, D))
EM final NLL = 1.4213513465550092

julia> println("G_em = $G_em")
G_em = gaussian 1 0.29810591081721827 1.0048372887886312
```

Similarly to the example above, we create a `Gaussian Node` with both `mean` and `variance` differente from the distribution `gauss`. However, we apply the Gradient Descent algorithm in the learning process.

```julia
julia> G_grad = RPCircuits.Gaussian(1, -0.15, 2.5);

julia> L_grad = Gradient(G_grad, gauss=true);

julia> println("Grad initial NLL = ", NLL(G_grad, D))
Grad initial NLL = 1.6182111383613536

julia> for i = 1:2_500
           update(L_grad, D; learningrate=0.01, learngaussians=true, verbose=false)
       end

julia> println("Grad final NLL = ", NLL(G_grad, D))
Grad final NLL = 1.4536896433734792

julia> println("G_grad = $G_grad")
G_grad = gaussian 1 0.29543688549190184 1.4738744413443041