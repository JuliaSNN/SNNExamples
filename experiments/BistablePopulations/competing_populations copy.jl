using Revise
using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Random
##

# Define each of the network recurrent assemblies
function attractor_model(istdp; N=400, n_assemblies=3, a=1)
    # Number of neurons in the network
    function get_subnet(N, name, istdp, a)
        N = N
        # Create dendrites for each neuron
        E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -55mV, At = 1mV, a=0, b=0), name="Exc_$name")
        # Define interneurons 
        I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV), name="Inh_$name")
        # Define synaptic interactions between neurons and interneurons
        E_to_I = SNN.SpikingSynapse(E, I, :he, p = 0.2, μ = 1.0, name="E_to_I_$name")
        E_to_E = SNN.SpikingSynapse(E, E, :he, p = 0.2, μ = a * 0.5, name="E_to_E_$name")
        I_to_I = SNN.SpikingSynapse(I, I, :hi, p = 0.2, μ = 1.0, name="I_to_I_$name")
        I_to_E = SNN.SpikingSynapse(
            I,
            E,
            :hi,
            p = 0.2,
            μ = 20,
            param = istdp,
            name="I_to_E_$name",
        )
        norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 10ms))

        # Store neurons and synapses into a dictionary
        pop = SNN.@symdict E I
        syn = SNN.@symdict I_to_E E_to_I E_to_E norm I_to_I
        noise = SNN.PoissonStimulus(E, :he, param=2.5kHz, cells=:ALL)
        # Return the network as a tuple
        SNN.monitor([E, I], [:fire])
        SNN.monitor(I_to_E, [:W], sr=10Hz)
        SNN.merge_models(pop, syn, noise=noise, silent=true)
    end

    subnets = Dict(Symbol("sub_$n") => get_subnet(N, n, istdp, a) for n = 1:n_assemblies)
    syns = Dict{Symbol,Any}()
    for i in eachindex(subnets)
        for j in eachindex(subnets)
            i == j && continue
            symbolEI = Symbol(string("$(i)E_to_$(j)I_lateral"))
            synapseEI = SNN.SpikingSynapse(subnets[i].pop.E, subnets[j].pop.I, :he, p = 0.2, μ = 1.5,)
            push!(syns, symbolEI => synapseEI)
            symbolEE = Symbol(string("$(i)E_to_$(j)E_lateral"))
            synapseEE = SNN.SpikingSynapse( subnets[i].pop.E, subnets[j].pop.E, :he, p = 0.2, μ = 0.2)
            push!(syns, symbolEE => synapseEE)
        end
    end
    SNN.merge_models(subnets, syns,)
end

function add_stimulus(network, population, interval)
    trig_param = PoissonStimulusInterval(fill(0.8kHz, 400), [interval])
    name = Symbol("stim_$(population)_$(randstring(2))")
    trigger = Dict{Symbol,Any}(
        name => SNN.PoissonStimulus(getfield(network.pop,population), :he, param=trig_param, cells=:ALL, name=string(name)),
    )
    SNN.merge_models(network, trigger, silent=true)
end

n_assemblies = 4
function test_istdp(istdp; a=1) 
    network = attractor_model(istdp, a=a)
    train!(model = network, duration = 5000ms, pbar = true, dt = 0.125)
    clear_records(network.pop)
    clear_records(network.syn)
    model = add_stimulus(network, :sub_1_E, [5s, 7s])
    # model = add_stimulus(model, :sub_2_E, [10s, 12s])
    train!(model = model, duration = 15000ms, pbar = true, dt = 0.125)
    return model
end

#
function iSTDP_activity(network, istdp; interval= 1s:20ms:15s)
    i_to_e = SNN.filter_items(network.syn, condition=p->occursin("I_to_E", p.name))
    w_i = map(eachindex(i_to_e)) do i
        w, r_t = record(i_to_e[i], :W, interpolate=true)
        mean(w, dims=1)[1,:]
    end |> collect

    _, r_t= record(i_to_e[1], :W, interpolate=true)
    p1 = plot(r_t./1000, w_i, xlabel="Time (s)", ylabel="Synaptic weight", legend=:topleft, title="I to E synapse", labels=["pop 1" "pop 2" "pop 3" "pop 4"], lw=4)

    Epop = SNN.filter_items(network.pop, condition=p->occursin("E", p.name))
    rates, interval = SNN.firing_rate(Epop, interval=interval, interpolate=false)
    rates = mean.(rates)
    p2 = plot(interval./1000, rates, xlabel="Time (s)", ylabel="Firing rate (Hz)", legend=:topleft, title="Firing rate of the exc. pop", lw=4, labels= ["pop 1" "pop 2" "pop 3" "pop 4"])#, yscale=:log10, ylims=(0.1,50))
    p3 = SNN.stdp_kernel(istdp, fill=false)
    p4 = SNN.raster(network.pop, interval, every=3)
    # plot(p3, p1, p4, p2, layout=(2,2), size=(800,800), margin=5Plots.mm)
    plot(p4, p2, layout=(2,1), size=(800,800), margin=5Plots.mm)
end



path = mkpath(plotsdir("iSTDP_lateralExc"))
istdp = SNN.iSTDPParameterRate(τy = 20ms, η = 0.5, r=5Hz) 
modelRate = test_istdp(istdp, a=1)
p = iSTDP_activity(modelRate, istdp)
savefig(p, joinpath(path,"iSTDP_rate.png"))

istdp = SNN.iSTDPParameterTime(τy = 20ms, η = 0.5) 
@profview modelTime = test_istdp(istdp, a=1)
p = iSTDP_activity(modelTime, istdp)
savefig(p, joinpath(path,"iSTDP_time.png"))


Alearn = 1e-2
istdp= AntiSymmetricSTDP(   A_x = Alearn*1e3,
                            A_y  = Alearn*1e3,
                            αpre = 0,#-0.5f0Alearn,
                            αpost = 0.5f0Alearn,
                            τ_x = 15ms,
                            τ_y = 60ms,
                            Wmax = 80
                           )
stdp_kernel(istdp)
modelAntiSym = test_istdp(istdp, a=2)
p = iSTDP_activity(modelAntiSym, istdp)
savefig(p, joinpath(path,"iSTDP_antisym.png"))
p
Alearn = 1e-2
istdp= AntiSymmetricSTDP(   A_x = -Alearn*1e3,
                            A_y  = -Alearn*1e3,
                            αpre = 0f0,
                            αpost = 0.05Alearn,
                            τ_x = 15ms,
                            τ_y = 60ms,
                            Wmax = 80
                           )
stdp_kernel(istdp)
modelAntiSym_anthebbian = test_istdp(istdp, a=1)
p = iSTDP_activity(modelAntiSym_anthebbian, istdp)
savefig(p, joinpath(path,"iSTDP_antisym_antiheb.png"))
##

Alearn= 1e-2
istdp = SymmetricSTDP( A_x = 1e3*Alearn,
                            A_y = 1e3*Alearn,
                            αpre = -0.5Alearn,
                            αpost = 0.0Alearn,
                            τ_x = 30ms,
                            τ_y = 600ms,
                            Wmax = 80,
                           )
modelHebbian = test_istdp(istdp, a=2)
iSTDP_activity(model, istdp)

# istdp = STDPParameter(A_pre =5e-1, 
#                            A_post=5e-1,
#                            τpre  =25ms,
#                            τpost =25ms)
# Instantiate the network assemblies and local inhibitory populations
# Add noise to each assembly
# Create synaptic connections between the assemblies and the lateral inhibitory populations



# Merge the models and run the simulation, the merge_models function will return a model object (syn=..., pop=...); the function has strong type checking, see the documentation.

# network = SNN.merge_models(network, trigger=trigger)

# Define a time object to keep track of the simulation time, the time object will be passed to the train! function, otherwise the simulation will not create one on the fly.
# train!(model = network, duration = 5000ms, pbar = true, dt = 0.125)


##

# define the time interval for the analysis
# select only excitatory populations
# get the spiketimes of the excitatory populations and the indices of each population
exc_populations = SNN.filter_items(network.pop, condition=p->occursin("Exc", p.name))
exc_spiketimes = SNN.spiketimes(network.pop)
# exc_indices = SNN.population_indices(exc_populations)
# calculate the firing rate of each excitatory population
rates, intervals = SNN.firing_rate(exc_populations, interval = interval,  τ = 50, interpolate=false)
rates = mean.(rates)

# Plot the firing rate of each assembly and the correlation matrix
p1 = plot()
for i in eachindex(rates)
    plot!(
        interval,
        rates[i],
        label = "Assembly $i",
        xlabel = "Time (ms)",
        ylabel = "Firing rate (Hz)",
        xlims = (2_000, 15_000),
        legend = :topleft,
    )
end
plot!()


##

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
