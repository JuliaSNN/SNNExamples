using Revise
using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Random
##


##
attractor = 5
Alearn= 1e-3
Ebase = 0.02

synapse_nmda = let
    SomaGlu = Glutamatergic(
        Receptor(E_rev = 0.0, τr = 1ms, τd = 6.0ms, g0 = 0.7),
        ReceptorVoltage(E_rev = 0.0, τr = 1ms, τd = 100.0, g0 = 0.15, nmda = 1.0f0),
    )
    SomaGABA = GABAergic(
        Receptor(E_rev = -70.0, τr = 0.5, τd = 2.0, g0 = 2.),
        Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.06), # τd = 100.0
    )
    SomaNMDA = NMDAVoltageDependency()
    SomaSynapse = Synapse(SomaGlu, SomaGABA)
end
synapse_ampa = let
    SomaGlu = Glutamatergic(
        Receptor(E_rev = 0.0, τr = 1ms, τd = 6.0ms, g0 = 2.),
        ReceptorVoltage( nmda = 0.0f0),
    )
    SomaGABA = GABAergic(
        Receptor(E_rev = -70.0, τr = 0.5, τd = 2.0, g0 = 2.),
        Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.06), # τd = 100.0
    )
    SomaNMDA = NMDAVoltageDependency()
    SomaSynapse = Synapse(SomaGlu, SomaGABA)
end
##
config = (
            intervals = [[3s, 4s], [6s, 7s]],
            E_to_I = (p = 0.2, μ = 1.0),
            E_to_E = (p = 0.2, μ = attractor * Ebase), # was 0.5
            I_to_I = (p = 0.2, μ = 1.0),
            I_to_E = (p = 0.2, μ = 20, param = istdp),
            lateral_EI = (p = 0.2, μ = 2.5),
            lateral_EE = (p = 0.2, μ = Ebase),
            N= 400,
            n_assemblies =2,
            duration = 10s,
            # adex = SNN.AdExSynapseParameter(synapse_nmda; Vr = -55mV, At = 1mV, a=0, b=0, ),
            # path = mkpath(plotsdir("iSTDP_NMDA_lateralExc")),
            adex = SNN.AdExSynapseParameter(synapse_ampa; Vr = -55mV, At = 1mV, a=0, b=0, ),
            path = mkpath(plotsdir("iSTDP_lateralExc")),
            noise = 2.5kHz,
            ext_stim = 0.5kHz,
        )
istdps = (
    istdp_rate = SNN.iSTDPParameterRate(τy = 20ms, η = Alearn*1e3, r=5Hz) ,
    istdp_time = SNN.iSTDPParameterTime(τy = 20ms, η = Alearn*1e3) ,
    Symmetrical= AntiSymmetricSTDP(   A_x = Alearn*1e3,
                                A_y  = Alearn*1e3,
                                αpre = 0,#-0.5f0Alearn,
                                αpost = 0.5f0Alearn,
                                τ_x = 15ms,
                                τ_y = 60ms,
                                Wmax = 200
                            ),
    antiHebbian= AntiSymmetricSTDP(   A_x = -Alearn*1e3,
                                A_y  = -Alearn*1e3,
                                αpre = 0f0,
                                αpost = 0.05Alearn,
                                τ_x = 15ms,
                                τ_y = 60ms,
                                Wmax = 200
                            ),
    Hebbian = SymmetricSTDP( A_x = 1e3*Alearn,
                                A_y = 1e3*Alearn,
                                αpre = -0.5Alearn,
                                αpost = 0.0Alearn,
                                τ_x = 30ms,
                                τ_y = 600ms,
                                Wmax = 200
                            )
)
##

models = Dict()
for name in keys(istdps)
    istdp = istdps[name]
    config = (config..., istdp = istdp)
    model = test_istdp(config)
    fr, r = firing_rate(model.pop.sub_1_E, interval = 1s:3s)
    @show "Firing rate:" mean(fr)
    p = iSTDP_activity(model, istdp, config)
    savefig(p, joinpath(config.path,"iSTDP_$(name).pdf"))
    push!(models, name => model)
end

##
istdp = istdps[:istdp_rate]
config = (config..., istdp = istdp)
p = iSTDP_activity(model, istdp, config)
model = test_istdp(config)
models[:istdp_rate] = model
raster(models[:istdp_rate].pop, 0s:10s)
models[:istdp_rate].syn.sub_1_I_to_E.W
W, r = record(models[:istdp_rate].syn.sub_1_I_to_E, :W, interpolate=true)
plot(r, mean(W, dims=1)[1,:], xlabel = "Time (ms)", ylabel = "Synaptic weight", legend = false)

##

##


# Merge the models and run the simulation, the merge_models function will return a model object (syn=..., pop=...); the function has strong type checking, see the documentation.

# network = SNN.merge_models(network, trigger=trigger)

# Define a time object to keep track of the simulation time, the time object will be passed to the train! function, otherwise the simulation will not create one on the fly.
# train!(model = network, duration = 5000ms, pbar = true, dt = 0.125)


##

# # define the time interval for the analysis
# # select only excitatory populations
# # get the spiketimes of the excitatory populations and the indices of each population
# exc_populations = SNN.filter_items(network.pop, condition=p->occursin("Exc", p.name))
# exc_spiketimes = SNN.spiketimes(network.pop)
# # exc_indices = SNN.population_indices(exc_populations)
# # calculate the firing rate of each excitatory population
# rates, intervals = SNN.firing_rate(exc_populations, interval = interval,  τ = 50, interpolate=false)
# rates = mean.(rates)

# # Plot the firing rate of each assembly and the correlation matrix
# p1 = plot()
# for i in eachindex(rates)
#     plot!(
#         interval,
#         rates[i],
#         label = "Assembly $i",
#         xlabel = "Time (ms)",
#         ylabel = "Firing rate (Hz)",
#         xlims = (2_000, 15_000),
#         legend = :topleft,
#     )
# end
# plot!()


# ##

# cor_mat = zeros(length(rates), length(rates))
# for i in eachindex(rates)
#     for j in eachindex(rates)
#         cor_mat[i, j] = cor(rates[i], rates[j])
#     end
# end
# p2 = heatmap(
#     cor_mat,
#     c = :bluesreds,
#     clims = (-1, 1),
#     xlabel = "Assembly",
#     ylabel = "Assembly",
#     title = "Correlation matrix",
#     xticks = 1:3,
#     yticks = 1:3,
# )
# plot(p1, p2, layout = (2, 1), size = (600, 800), margin = 5Plots.mm)
