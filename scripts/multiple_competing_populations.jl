using Revise
using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Plots
using Statistics

# Define each of the network recurrent assemblies
function define_network(N, name)
    # Number of neurons in the network
    N = N
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -55mV, At = 0mV, b=0, a=0), name="Exc_$name")
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV), name="Inh_$name")
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :he, p = 0.2, μ = 1.0)
    E_to_E = SNN.SpikingSynapse(E, E, :he, p = 0.2, μ = 0.5)#, param = SNN.vSTDPParameter())
    I_to_I = SNN.SpikingSynapse(I, I, :hi, p = 0.2, μ = 1.0)
    I_to_E = SNN.SpikingSynapse(
        I,
        E,
        :hi,
        p = 0.2,
        μ = 1,
        # param = SNN.iSTDPParameterRate(r = 4Hz),
    )
    norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 10ms))

    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I)
    syn = dict2ntuple(@strdict I_to_E E_to_I E_to_E norm I_to_I)
    # Return the network as a tuple
    noise = ExcNoise(E, μ = 15.8f0)
    SNN.monitor([E, I], [:fire])
    (pop = pop, syn = syn)
end

n_assemblies = 4
## Instantiate the network assemblies and local inhibitory populations
subnets = Dict(Symbol("sub_$n") => define_network(400, n) for n = 1:n_assemblies)
# Add noise to each assembly
noise = Dict(Symbol("noise_$(i)") => SNN.PoissonStimulus(subnets[i].pop.E, :he, param=4.5kHz, cells=:ALL) for i in eachindex(subnets))
# Create synaptic connections between the assemblies and the lateral inhibitory populations
syns = Dict{Symbol,Any}()
for i in eachindex(subnets)
    for j in eachindex(subnets)
        i == j && continue
        push!(
            syns,
            Symbol("lateral_$(i)E_to_$(j)I") => SNN.SpikingSynapse(
                subnets[i].pop.E,
                subnets[j].pop.I,
                :he,
                p = 0.2,
                μ = 10.25,
            ),
        )
    end
end

## Merge the models and run the simulation, the merge_models function will return a model object (syn=..., pop=...); the function has strong type checking, see the documentation.
network = SNN.merge_models(noise, subnets, syns)

# Define a time object to keep track of the simulation time, the time object will be passed to the train! function, otherwise the simulation will not create one on the fly.
time_keeper = SNN.Time()
train!(model = network, duration = 5000ms, time = time_keeper, pbar = true, dt = 0.125)

time_keeper = SNN.Time()
SNN.clear_records([network.pop...])
train!(model = network, duration = 15000ms, time = time_keeper, pbar = true, dt = 0.125)

## Create a model object with only the populations to run the analysis
populations = network.pop

# Plot the raster plot of the network
SNN.raster([populations...], [12s, 15s])
##

# define the time interval for the analysis
interval = 0:20:SNN.get_time(time_keeper)

# select only excitatory populations
exc_populations = SNN.filter_populations(populations, :E)

# get the spiketimes of the excitatory populations and the indices of each population
spiketimes = SNN.spiketimes(exc_populations)
indices = SNN.population_indices(populations, :E)

# calculate the firing rate of each excitatory population
rates = map(eachindex(indices)) do i
    rates, intervals =
        SNN.firing_rate(spiketimes, interval = interval, pop = indices[i], τ = 50)
    mean_rate = mean(rates)
end

## Plot the firing rate of each assembly and the correlation matrix
p1 = plot()
for i in eachindex(rates)
    plot!(
        interval,
        rates[i],
        label = "Assembly $i",
        xlabel = "Time (ms)",
        ylabel = "Firing rate (Hz)",
        xlims = (10_000, 15_000),
        legend = :topleft,
    )
end
plot!()

cor_mat = zeros(length(rates), length(rates))
for i in eachindex(rates)
    for j in eachindex(rates)
        cor_mat[i, j] = cor(rates[i], rates[j])
    end
end
p2 = heatmap(
    cor_mat,
    c = :bluesreds,
    clims = (-1, 1),
    xlabel = "Assembly",
    ylabel = "Assembly",
    title = "Correlation matrix",
    xticks = 1:3,
    yticks = 1:3,
)
plot(p1, p2, layout = (2, 1), size = (600, 800), margin = 5Plots.mm)
