using Plots
using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils

# Define the network
network = let
    # Number of neurons in the network
    N = 1000
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV))
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV))
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :ge, p = 0.2, μ = 1.0)
    E_to_E = SNN.SpikingSynapse(E, E, :ge, p = 0.2, μ = 0.5)#, param = SNN.vSTDPParameter())
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
    # Return the network as a tuple
    (pop = pop, syn = syn)
end

# Create background for the network simulation
noise = SNN.PoissonStimulus(network.pop.E, :ge, param=2.8kHz, cells=:ALL)

# Combine all
model = SNN.merge_models(network, noise=noise)

#
@info "Initializing network"
simtime = SNN.Time()
SNN.monitor([network.pop...], [:fire])

train!(model=model, duration = 25000ms, time = simtime, dt = 0.125f0, pbar = true)
##

plots = map(1:10) do i
    histogram(autocorrelogram(spiketimes(model.pop.E)[i],400ms), bins=-400:20:400)
end
plot(plots..., layout = (2,5), size = (800, 400), legend = false)

spiketimes = SNN.spiketimes(network.pop.E)
SNN.raster([network.pop...], [1s, 2s])
# SNN.vecplot(network.pop.E, [:ge], neurons = 1:10, r = 800ms:4999ms)
rates, intervals = SNN.firing_rate(network.pop.E, interval=0:10:15s, τ=20ms)
# plot(rates[1:100,1:end]', legend=false)
plot(intervals, mean(rates, dims = 1)[1, :], legend = false)
##

network.pop.E.records

## The simulation achieves > 1.0 iteration per second on my M1 Pro machine.
