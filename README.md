RPCircuits.jl
=============

A Julia package for learning/constructing [Probabilistic Circuits][probcirc20]. Particularly,
*RPCircuits* implements:

* Tools for building probabilistic circuit
* EM and gradient ascent parameter learning
* Structure learning for probabilistic circuits
* Probabilistic queries

**Note:** This repository contains everything you need to reproduce the paper "[Fast And Accurate
Learning of Probabilistic Circuits by Random Projections][link_artigo]" and LearnRP ("[Scalable
Learning of Probabilistic Circuits][link_msc]", Chapter 5).

[link_msc]: https://www.teses.usp.br/teses/disponiveis/45/45134/tde-23052022-122922/en.php
[link_artigo]: https://www.ime.usp.br/~renatolg/docs/geh21b_paper.pdf
[probcirc20]: http://starai.cs.ucla.edu/papers/ProbCirc20.pdf

## Quick Tutorial

You may either install `RPCircuits` with `julia -e 'using Pkg; Pkg.add("/path/to/RPCircuits")'`
or use it locally without installation, in which case you will need to `activate` `RPCircuits`
for every Julia session (see [Installation](#installation) for more info):

```julia
# Activate RPCircuits
using Pkg; Pkg.activate("/path/to/RPCircuits")

# We may now use RPCircuits as normal
using RPCircuits

```

### Manually building circuits

We begin by creating four indicator leaf nodes.

```julia
julia> a, b, na, nb = Indicator(1, 1.0), Indicator(2, 1.0), Indicator(1, 0.0), Indicator(2, 0.0)
(indicator 1 1.0, indicator 2 1.0, indicator 1 0.0, indicator 2 0.0)
```
The `Indicator` function takes 2 positional arguments and a keyword argument. The first two
arguments are: the `index` (essentially the identifier) of the variable, and the `value` such that
the node outputs `true` when the variable is set to `value`. The keyword argument `tolerance` sets
a maximum discrepancy when evaluating the indicator at a given value (its default value is
`1e-6`.).

As an indicator function, we can evaluate each node on various inputs.
```julia
julia> a(1,1), b(0,0), na(0,1), nb(1,0)
(1.0, 0.0, 1.0, 1.0)
```

Since the purpose of this package is to enable construction and learning of Probabilistic, we show
next how to create `product` and `sum` nodes, and how to build circuits from a bottom-up approach.
First, we have four product nodes with all combinations of `a`, `b` and their negations.
```julia
julia> P1, P2, P3, P4 = Product([a,b]), Product([a,nb]), Product([na,b]), Product([na,nb])
(* 1 2, * 1 2, * 1 2, * 1 2)
```
The `Product` function takes a vector `v = [v1,..., vn]` of nodes as inputs, and creates a product
node `P` with children `v1,...,vn`.

Next, we create a sum node `S` whose children are `P1`, `P2`, `P3` and `P4`, and has respective
weights `0.25`, `0.25`, `0.25`, `0.25`.
```julia
julia> S = Sum([P1,P2,P3,P4], [0.25,0.25,0.25,0.25])
Circuit with 9 nodes (1 sum, 4 products, 4 leaves) and 2 variables:
  1 : + 1 0.25 2 0.25 3 0.25 4 0.25
  2 : * 1 2
  3 : * 1 2
  4 : indicator 1 0.0
  5 : * 1 2
  6 : indicator 2 0.0
  7 : * 1 2
  8 : indicator 2 1.0
  9 : indicator 1 1.0
```

The `Sum` functions takes two arguments: a vector of nodes (i.e., the `children`), and a vector of
the respective `weights`.

### Learning the parameters of circuits

One of the main purposes of this package is learning of Probabilistic Circuits. We first start out
with a fixed structure and learn its sum weights. Taking the same structure as before as a toy
example, let's set the sum weights to `weights = [0.4, 0.3, 0.2, 0.1]` and call this new circuit
the *target* distribution `f`. We next sample from this circuit to produce a dataset.
```julia
using Random

Random.seed!(42) # Locking seed

# Construct the structure of the target distribution
a, na, b, nb = Indicator(1, 1.0), Indicator(1, 0.), Indicator(2, 1.), Indicator(2, 0.)
P1, P2, P3, P4 = Product([a,b]), Product([a,nb]), Product([na,b]), Product([na,nb])
f = Sum([P1, P2, P3, P4], [0.4, 0.3, 0.2, 0.1])

# Sample N samples from it
N = 1_000
D = rand(f, N)
```

We shall now build a new circuit `g` whose structure is the same as `f` but whose weights are set
to uniform. Ideally, when learning from the dataset `D`, we want `g`'s weights to change to be
(approximately) equal to `f`'s.
```julia
# Let's copy the same structure as f, but set its weights to a uniform
g = copy(f)
g.weights .= [0.25,0.25,0.25,0.25]
```

Now that we have our distribution `g`, we are ready to fit it to the data `D`. We'll do this by
[Expectation-Maximization][em-spns], maximizing the log-likelihood (or equivalently minimizing the
negative log-likelihood, here denoted by `NLL`).

[em-spns]: https://ipa.iwr.uni-heidelberg.de/ipabib/Papers/Desana2016.pdf
```julia
println("Target model NLL = ", NLL(f, D))
println("Initial circuit NLL = ", NLL(g, D))
L = SEM(g) # Learner with EM algorithm
for i = 1:50
    update(L, D) # Iterations of the EM algorithm
end
println("Final circuit NLL = ", NLL(g, D))
println("          weights = ", round.(g.weights, digits = 2))
```
```julia
Target model NLL = 1.2905776805822866
Initial circuit NLL = 1.3862943611198644
Final circuit NLL = 1.2891629841331291
          weights = [0.38, 0.32, 0.19, 0.11]
```

## Installation

If you're not familiar with Julia's [REPL][repl_doc] or [Pkg][pkg_doc], we highly recommend having
a look at the linked Julia docs. To install all dependencies,

1. Install [Julia][julialang] version 1.7.2

2. Clone this repository
   ```
   git clone https://github.com/RenatoGeh/RPCircuits.jl
   ```

3. Start the [Julia REPL][repl_doc] using the command `julia`. Next, switch to package mode by
   entering the command `]`. Your screen should look like

   ```julia
   (@v1.7) pkg>
   ```

4. Within [Pkg mode][pkg_doc], activate the RPCircuits environment via 
   ```julia
   activate /path/to/RPCircuits
   ```
   and install all dependencies with
   ```julia
   instantiate
   ```

5. You may then locally install `RPCircuits` with
   ```julia
   add /path/to/RPCircuits
   ```

## Troubleshooting

If you have any problems installing the [`BlossomV`][blossomv] package dependency, try installing
the latest version of [`gcc`][gcc] **and** `g++` (more info [here][blossomv_build]). Once
`g++` has been installed, build `BlossomV`, through `build BlossomV` within Julia's `Pkg` mode.
Similarly, if you have any problems installing `HDF5`, `MAT` or `MLDatasets`, try building each
package one at a time.

**Note:** If you use an Arch based Linux distribution, you may want to install `base devel`. In
case you run into any other problems `Arpack`, `GaussianMixtures`, `HDF5`, we highly recommend
switching to `julia-bin` via AUR and reinstalling `RPCircuits`.

[julialang]: https://julialang.org/
[repl_doc]: https://docs.julialang.org/en/v1/stdlib/REPL/
[pkg_doc]: https://pkgdocs.julialang.org/v1
[blossomv]: https://github.com/mlewe/BlossomV.jl
[blossomv_build]: https://github.com/mlewe/BlossomV.jl#building
[gcc]: https://gcc.gnu.org/

## How to cite

To acknowledge this package, please cite:
```
@mastersthesis{geh22a,
  author = {Renato Lui Geh},
  title  = {Scalable Learning of Probabilistic Circuits},
  school = {University of S{\~{a}}o Paulo},
  type   = {Master's in Computer Science dissertation},
  year   = {2022},
  month  = {April},
  doi    = {10.11606/D.45.2022.tde-23052022-122922},
  url    = {https://doi.org/10.11606/D.45.2022.tde-23052022-122922}
}

@inproceedings{geh2021fast,
   title={Fast And Accurate Learning of Probabilistic Circuits by Random Projections},
   author={Renato Geh and Denis Mau{\'a}},
   booktitle={The 4th Workshop on Tractable Probabilistic Modeling},
   year={2021},
   url={https://www.ime.usp.br/~renatolg/docs/geh21b_paper.pdf}
}
```

