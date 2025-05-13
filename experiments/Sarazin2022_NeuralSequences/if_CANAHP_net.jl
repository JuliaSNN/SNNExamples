using Plots
using SpikingNeuralNetworks
using Distributions
SNN.@load_units

IF_param = IF_CANAHPParameter()
E = SNN.IF_CANAHP(N=450, param=IF_param, name="Excitatory")
I = SNN.IF_CANAHP(N=150, param=IF_param, name="Inhibitory")
# SNN.monitor!(E, [:v, :fire, :hi, :he, :g, :h, :I, :syn_curr], sr = 8000Hz)

I_param_E = SNN.CurrentNoiseParameter(E.N; I_base=100pA, I_dist=Normal(450pA, 100pA), α=1. )
I_param_I = SNN.CurrentNoiseParameter(E.N; I_base=100pA, I_dist=Normal(450pA, 100pA), α=1. )
I_stimE = SNN.CurrentStimulus(E, param=I_param_E)
I_stimI = SNN.CurrentStimulus(I, param=I_param_I)


synapses = (
    E_to_E = SNN.SpikingSynapse(E, E, :he, p=0.2, μ=0.01, name="E_to_E"),
    E_to_I = SNN.SpikingSynapse(E, I, :he, p=0.2, μ=0.01, name="E_to_I"),
    I_to_E = SNN.SpikingSynapse(I, E, :hi, p=0.2, μ=0.1, name="I_to_E"),
    I_to_I = SNN.SpikingSynapse(I, I, :hi, p=0.2, μ=0.1, name="I_to_I"),
)

model = merge_models(;E, I_stimE, I_stimI, I, synapses, silent = true)
SNN.monitor!(model.pop, [:v, :fire, :v], sr = 200Hz)

#
SNN.sim!(;model, duration = 15s, pbar = true)
# synapses.E_to_E.W
#
raster(model.pop, 5s:10s)
##
fr, r, labels = SNN.firing_rate(model.pop, interval=5s:10s, pop_average=true, τ=10ms)
plot(fr)

## 
vecplot(model.pop.E, :v, interval=5s:10s, neurons=8, title="E", xlabel="Time (s)", ylabel="Membrane potential (mV)", xlims=(0, 5), ylims=(-80, 50))

using PyCall
rastermap =pyimport("rastermap")

spikes, r = bin_spiketimes(spiketimes(model.pop.E), time_range=5s:10ms:10s, do_sparse=false)

scatter(mean(spikes, dims=2))
# , time_range=5s:10ms:10s, sparse=false)
# sample = bin_spiketimes(spike_times[1]; kwargs...)