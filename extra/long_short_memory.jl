# E/I network of 1000 AdEx and 250 IF neurons with short-term plasticity and voltage-based STDP on E -> E, rate-based
# inhibitory STDP on I -> E and additive synaptic normalisation, driven by Poisson input for 5 s. Plots a raster, the
# voltages of ten excitatory neurons and the mean excitatory rate. A scratch test of long- and short-term plasticity
# together; see tutorials/plasticity/AdEx_plasticity.jl and tutorials/networks/AdEx_STP.jl for the maintained versions.
# Runtime: about 55 s.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie, Statistics

N = 1000
E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV), name = "E")
I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "I")
E_to_I = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 3.0))
E_to_E = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5), LTPParam = SNN.vSTDPParameter(),
                            STPParam = SNN.MarkramSTPParameter())
I_to_I = SNN.SpikingSynapse(I, I, :gi; conn = (p = 0.2, μ = 4.0))
I_to_E = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 1), LTPParam = SNN.iSTDPRate(r = 4Hz))
norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 30ms), [E_to_E])
noise = SNN.PoissonStimulus(E, :ge; param = SNN.PoissonFixed(rate = 2.8kHz), neurons = :ALL)
model = SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, norm, noise, silent = true)

SNN.monitor!(model.pop, [:fire])
SNN.monitor!(E, [:v], sr = 1kHz)
SNN.train!(model = model, duration = 5000ms, dt = 0.1f0)

fig = Figure(size = (800, 900))
SNN.raster!(Axis(fig[1, 1], ylabel = "Neuron"), model.pop, 4s:5s)
ax2 = Axis(fig[2, 1], ylabel = "V (mV)")
SNN.vecplot!(ax2, E, :v; neurons = 1:10, interval = 800ms:1ms:5s)
rates, intervals = SNN.firing_rate(E, interval = 0:10ms:5s, τ = 100ms)
ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "Rate (Hz)")
lines!(ax3, intervals ./ 1000, vec(mean(rates, dims = 1)))
fig
