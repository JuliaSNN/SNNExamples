# Test of voltage-based STDP on one synapse: in an E/I network of 1000 AdEx and 250 IF neurons (inhibitory STDP and
# additive normalisation) a single extra E -> E synapse from neuron 23 to neuron 58 with weight 5 and a vSTDP rule
# is added; the weights of the extra synapse are recorded while the network is driven by Poisson input for 5 s.
# Plots a raster and the mean excitatory rate, and the weight of the test synapse over time. The network activity
# collapses after an initial burst of 0.5 s, so the weight changes only during the burst.
# Runtime: about 55 s.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie, Statistics

N = 1000
E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV), name = "E")
I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "I")
E_to_I = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 3.0))
E_to_E = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5))
I_to_I = SNN.SpikingSynapse(I, I, :gi; conn = (p = 0.2, μ = 4.0))
I_to_E = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 1), LTPParam = SNN.iSTDPRate(r = 4Hz))
norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 30ms), [E_to_E])
noise = SNN.PoissonStimulus(E, :ge; param = SNN.PoissonFixed(rate = 2.8kHz), neurons = :ALL)

cellA = 23
cellB = 58
W = zeros(E.N, E.N)           # N_post x N_pre
W[cellB, cellA] = 5
measured_syn = SNN.SpikingSynapse(E, E, :ge; conn = W, LTPParam = SNN.vSTDPParameter(), name = "measured")

model = SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, norm, noise, measured_syn, silent = true)
@info "Initializing network"
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(measured_syn, [:W], sr = 100Hz)
SNN.train!(model = model, duration = 5000ms, dt = 0.125f0)

fig = Figure(size = (800, 800))
SNN.raster!(Axis(fig[1, 1], ylabel = "Neuron"), model.pop, 1s:2s)
rates, intervals = SNN.firing_rate(E, interval = 0:10ms:5s, τ = 100ms)
ax2 = Axis(fig[2, 1], ylabel = "Mean rate (Hz)")
lines!(ax2, intervals ./ 1000, vec(mean(rates, dims = 1)))
_, r = SNN.record(measured_syn, :W, range = true)
ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "W of the test synapse")
lines!(ax3, r ./ 1000, vec(SNN.getvariable(measured_syn, :W)[1, :]))
fig
