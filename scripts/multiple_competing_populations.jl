using DrWatson
@quickactivate "network_models"
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics

# Define each of the network recurrent assemblies
function define_network(N=800)
    # Number of neurons in the network
    N = N
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV))
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV))
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :ge, p = 0.2, σ = 0.0)
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
    SNN.monitor([E, I], [:fire])
    network = (pop = pop, syn = syn)
end

## Instantiate the network assemblies and local inhibitory populations
subnets = Dict("network$n" => define_network(200) for n in 1:10) 
# Add noise to each assembly
noise =Dict("$(i)_noise"=>ExcNoise(subnets[i].pop.E, σ = 10.8f0) for i in eachindex(subnets))
# Create synaptic connections between the assemblies and the lateral inhibitory populations
syns = Dict{String, Any}()
for i in eachindex(subnets)
    for j in eachindex(subnets) 
        i == j && continue
        push!(syns, "$(i)E_to_$(j)I" =>
            SNN.SpikingSynapse(subnets[i].pop.E, subnets[j].pop.I, :ge, p = 0.2, σ = 20.25))
    end
end

## Merge the models and run the simulation
network = SNN.merge_models(noise, subnets, syn = syns)
time_keeper = SNN.Time() 
train!(model=network, duration = 15000ms, time=time_keeper, pbar=true, dt=0.125)

## Create a model object with only the populations to run the analysis
populations = SNN.merge_models(subnets).pop
SNN.raster([populations...], [14s, 15s])

interval = 0:20:SNN.get_time(time_keeper)
indices = SNN.population_indices(populations, :E)
exc_populations = SNN.filter_populations(populations, :E)
spiketimes = SNN.spiketimes(exc_populations)

rates = map(eachindex(indices)) do i
    rates, intervals = SNN.firing_rate(spiketimes, interval , pop = indices[i], τ = 50)
    mean_rate = mean(rates)
end

plot()
for i in eachindex(rates)
    plot!(interval, rates[i], label = "Network $i")
end
plot!()

cor_mat = zeros(length(rates), length(rates))
for i in eachindex(rates)
    for j in eachindex(rates)
        cor_mat[i, j] = cor(rates[i], rates[j])
    end
end
heatmap(cor_mat, c=:bluesreds, clims=(-1, 1))


