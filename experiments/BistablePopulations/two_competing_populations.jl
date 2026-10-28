# Two recurrent E/I networks coupled by lateral excitation onto each other's inhibitory population (competition
# through the inhibitory neurons). Each network has 800 AdEx excitatory and 200 IF inhibitory neurons, inhibitory
# STDP (iSTDPRate) on I -> E, additive synaptic normalisation on E -> E, and Poisson background input. The
# script runs 15 s, plots a raster of both networks over the last second, and the correlation of the rates.
# Runtime: about 2.5 min.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie
using Statistics

# Define the network
function define_network(N = 800)
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV), name = "E")
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "I")
    E_to_I = SNN.SpikingSynapse(E, I, :he; conn = (p = 0.2, μ = 3.0))
    E_to_E = SNN.SpikingSynapse(E, E, :he; conn = (p = 0.2, μ = 0.5))
    I_to_I = SNN.SpikingSynapse(I, I, :hi; conn = (p = 0.2, μ = 1.0))
    I_to_E = SNN.SpikingSynapse(I, E, :hi; conn = (p = 0.2, μ = 10.0), LTPParam = SNN.iSTDPRate(r = 4Hz))
    norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 10ms), [E_to_E])
    return SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, norm, silent = true)
end

network1 = define_network(800)
network2 = define_network(800)
inter = (
    E1_to_I2 = SNN.SpikingSynapse(network1.pop.E, network2.pop.I, :he; conn = (p = 0.2, μ = 4.25)),
    E2_to_I1 = SNN.SpikingSynapse(network2.pop.E, network1.pop.I, :he; conn = (p = 0.2, μ = 4.25)),
)
noise = (
    A = SNN.PoissonStimulus(network1.pop.E, :he; param = SNN.PoissonFixed(rate = 4.5kHz, μ = 1.7f0), neurons = :ALL),
    B = SNN.PoissonStimulus(network2.pop.E, :he; param = SNN.PoissonFixed(rate = 4.5kHz, μ = 1.7f0), neurons = :ALL),
)

model = SNN.compose(noise, inter; n1 = network1, n2 = network2, silent = true)

SNN.monitor!(model.pop, [:fire])
SNN.train!(model = model, duration = 15000ms, dt = 0.125f0)
SNN.raster(model.pop, [14s, 15s])

##
rate1, intervals = SNN.firing_rate(model.pop.n1_E, interval = 1s:10ms:15s, τ = 10ms, pop_average = true)
rate2, _ = SNN.firing_rate(model.pop.n2_E, interval = 1s:10ms:15s, τ = 10ms, pop_average = true)
fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Firing rate (Hz)",
          title = "Correlation: $(round(cor(rate1, rate2), digits = 2))")
lines!(ax, intervals ./ 1000, rate1, label = "Network 1")
lines!(ax, intervals ./ 1000, rate2, label = "Network 2")
axislegend(ax)
fig
