import Pkg
Pkg.activate("/home/jonasrlg/code/RPCircuits.jl/")
using RPCircuits
a, na, b, nb = Indicator(1, 1.0), Indicator(1, 0.), Indicator(2, 1.), Indicator(2, 0.)
P1, P2, P3, P4 = Product([a,b]), Product([a,nb]), Product([na,b]), Product([na,nb])
f = Sum([P1, P2, P3, P4], [0.1, 0.2, 0.3, 0.4]) # Generating function
S = Sum([P1, P2, P3, P4], [0.25, 0.25, 0.25, 0.25]) # Approximating function
ccircuit = RPCircuits.compile(RPCircuits.CCircuit, S);
C = ccircuit.C;
N = 10_000
D = rand(S, N)
V = Matrix{Float64}(undef, N, length(C));
LL = RPCircuits.mplogpdf!(V, C, D)
entr = entropy(S)
summap = RPCircuits.map_sum_index(C, ccircuit.S)
RPCircuits.expected_flows(C, summap, V)./N
τ = 1e-4
println("Initial: ", -NLL(S,D) + τ*entr)
runs = 10
for i ∈ 1:runs
	RPCircuits.entropy_regularization!(C, summap, V; entropy_reg=τ)
	entr = entropy(S)
	println("Iteration ", i, ": ", -NLL(S,D) + τ*entr)
end
S.weights

