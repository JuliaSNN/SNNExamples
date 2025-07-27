using Plots
using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils

##
# Define the network
network = let
    # Number of neurons in the network
    N = 1000
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV))
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV))
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :ge, p = 0.2, μ = 3.0)
    LSSPparam = SNN.LSSPParameter(;long=SNN.vSTDPParameter(), short=SNN.STPParameter())
    E_to_E = SNN.SpikingSynapse(E, E, :ge, p = 0.2, μ = 0.5, param = LSSPparam)
    I_to_I = SNN.SpikingSynapse(I, I, :gi, p = 0.2, μ = 4.0)
    I_to_E = SNN.SpikingSynapse(
        I,
        E,
        :gi,
        p = 0.2,
        μ = 1,
        param = SNN.iSTDPParameterRate(r = 4Hz),
    )
    norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 30ms))
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I)
    syn = dict2ntuple(@strdict I_to_E E_to_I E_to_E norm I_to_I)
    # Return the network as a tuple
    (pop = pop, syn = syn)
end

noise = SNN.PoissonStimulus(network.pop.E, :ge, param=2.8kHz, cells=:ALL)
model = SNN.merge_models(network=network, noise=noise)
SNN.monitor([model.pop...], [:fire, :v])
# SNN.monitor([network.syn.E_to_E], [:x])
# SNN.monitor([network.syn.E_to_E], [:v])
# SNN.monitor([network.syn.E_to_E], [:ρ])

simtime = SNN.Time()
train!(model=model, duration = 5000ms, time = simtime, dt = 0.1f0, pbar = true)


SNN.vecplot(model.pop.network_E, [:v], neurons = 1:10, r = 800ms:4999ms)

spiketimes = SNN.spiketimes(model.pop.network_E)
SNN.raster([model.pop...], [0s, 5s])
rates, intervals = SNN.firing_rate(network.pop.E, interval=0:10:5s, τ=100ms)
plot(intervals,rates[1])
plot(mean(rates, dims = 1)[1, :], legend = false)
##

network.pop.E.records

## The simulation achieves > 1.0 iteration per second on my M1 Pro machine.


