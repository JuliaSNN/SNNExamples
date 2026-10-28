# Feed-forward chain of three IF neurons: a current kick on neuron 1 propagates through single
# strong synapses (weight 50) from neuron n to n+1; the voltage of the first and last neuron is plotted.
# Runtime: about 30 s.
using SNNPlots
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie

N = 3
E = SNN.IF(; N = N)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (μ = 0.5, p = 0.8))
for n = 1:(N-1)
    SNN.connect!(EE, n, n + 1, 50)
end
E.I[1] = 30

SNN.monitor!(E, [(:v, [1, N])])
SNN.sim!([E], [EE]; duration = 100ms)
SNN.vecplot(E, :v)
