using DrWatson
using Revise
using Plots
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
    E_to_I = SNN.SpikingSynapse(E, I, :ge, p = 0.2, μ = 3.0)
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
    pop = SNN.@symdict E I
    syn = SNN.@symdict I_to_E E_to_I E_to_E norm I_to_I
    (pop = pop, syn = syn)
end

# Create background for the network simulation
noise  = PoissonStimulus(network.pop[:E], :ge, param=2.8kHz, neurons=:ALL)
noise2 = PoissonStimulus(network.pop[:E], :gi, param=3kHz, neurons=:ALL)
init_model = merge_models(network, noise=noise, noise2=noise2)

## Save the model. Important to `merge_model` before saving to maintain shared memory pointers.
name = "network"
info = (state=:test, )
path = datadir("test")
model_path = SNNsave(path=path, name=name, model=init_model, info=info, count=1, config=info)

## Load the model, run the simulation and store the results
model = SNNload(;path, name,info, count=1).model
SNN.monitor!(model.pop, [:fire])
train!(model=model, duration = 5000ms, dt = 0.125f0, pbar = true)
model_path = SNNsave(path=path, name=name, model=model, info=info, count=2, config=info)

## Load the model, remove a stimulus, run the simulation, and plot the results
init_model =SNNload(;path, name, info, count=1).model
nogi_model = remove_element(init_model, :noise2)
SNN.monitor!(nogi_model.pop, [:fire])
train!(model=nogi_model, duration = 5000ms, dt = 0.125f0, pbar = true)
## Load the model and plot the results

p1 = SNN.raster(model.pop, [1s, 2s], title="Strong inhibitory noise")
p2 = SNN.raster(nogi_model.pop, [1s, 2s], title="No inhibitory noise")
plot(p1,p2, layout=(2,1), size=(800,800))


