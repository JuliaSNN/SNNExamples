using DrWatson
@quickactivate "network_models"
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots

# Define the network
function define_network(N=800)
    # Number of neurons in the network
    N = N
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV))
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV))
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :ge, p = 0.2, σ = 3.0)
    E_to_E = SNN.SpikingSynapse(E, E, :ge, p = 0.2, σ = 0.5)#, param = SNN.vSTDPParameter())
    I_to_I = SNN.SpikingSynapse(I, I, :gi, p = 0.2, σ = 1.)
    I_to_E = SNN.SpikingSynapse(
        I,
        E,
        :gi,
        p = 0.2,
        σ = 1,
        param = SNN.iSTDPParameterRate(r = 4Hz),
    )
    norm = SNN.SynapseNormalization(E, [E_to_E], param=SNN.AdditiveNorm(τ=10ms))

    # Store neurons and synapses into a dictionary
    pop  = dict2ntuple(@strdict E I)
    syn  = dict2ntuple(@strdict I_to_E E_to_I E_to_E norm I_to_I)
    # Return the network as a tuple
    noise = ExcNoise(E, σ = 15.8f0)
    network = (pop = pop, syn = syn)
end

subnets = map(1:10) do i
    network = define_network(200)
end

for i in 1:10
    noise = ExcNoise(subnets[i].pop.E, σ = 10.8f0)
    syns = dict2ntuple(
        map(1:10) do j
        E1_to_I2 = SNN.SpikingSynapse(subnets[i].pop.E, subnets[j].pop.I, :ge, p = 0.2, σ = 20.25)#, param = SNN.vSTDPParameter())
    end)
    # merge_models( @strdict noise=noise inter=(syn=syns, pop=()))
end




    # simulation = merge_models(@strdict network noise)
    # no_noise = merge_models(@strdict network)
    # SNN.monitor([no_noise.pop...], [:fire])
    # train!(model=simulation, duration = 15000ms)
    # SNN.raster([no_noise.pop...], [10s, 15s])
noise1 = ExcNoise(network1.pop.E, σ = 10.8f0)
noise2 = ExcNoise(network2.pop.E, σ = 10.8f0)
E1_to_I2 = SNN.SpikingSynapse(network1.pop.E, network2.pop.I, :ge, p = 0.2, σ = 20.25)#, param = SNN.vSTDPParameter())
E2_to_I1 = SNN.SpikingSynapse(network2.pop.E, network1.pop.I, :ge, p = 0.2, σ = 20.25)#, param = SNN.vSTDPParameter())
inter=(syn=(dict2ntuple(@strdict E1_to_I2 E2_to_I1)), pop=())
simulation = merge_models(@strdict network1 noise1 network2 noise2 inter)
no_noise = merge_models(@strdict network1 network2 inter)

## @info "Initializing network"
SNN.monitor([no_noise.pop...], [:fire])
train!(model=simulation, duration = 15000ms)
SNN.raster([no_noise.pop...], [10s, 15s])

##
using Statistics
rate1 = SNN.firing_rate(no_noise.pop.network1_E, 100ms)
rate2 = SNN.firing_rate(no_noise.pop.network2_E, 100ms)
r1 = mean(rate1, dims = 1)[1,1:10:end]
r2 = mean(rate2, dims = 1)[1,1:10:end]
cor(r1, r2)
plot(r1, label = "Network 1", xlabel = "Time (s)", ylabel = "Firing rate (Hz)")
plot!(r2, label = "Network 2", title="Correlation: $(cor(r1, r2))")
plot!(xlims = (100, 500))
##

# SNN.vecplot(network.pop.E, [:ge], neurons = 1:10, r = 800ms:4999ms)

# collect(network.syn)

# # Create background for the network simulation
# ## Combine all
# cellA = 23
# cellB = 58
# W = zeros(network.pop.E.N, network.pop.E.N)
# W[cellB, cellA] = 5
# measured_syn = SNN.SpikingSynapse(
#     network.pop.E,
#     network.pop.E,
#     :ge,
#     w = W,
#     param = SNN.vSTDPParameter(),
# )

# ##

# ## The simulation achieves > 1.0 iteration per second on my M1 Pro machine.