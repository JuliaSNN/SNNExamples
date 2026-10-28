# Working memory by short-term synaptic facilitation (Mongillo, Barak & Tsodyks 2008, Science 319:1543).
# 10,000-neuron E/I network with two E assemblies; the assemblies are strengthened after 1.8 s, and a series of
# 100 ms current pulses to the E population reactivates the stored item. Plots the assembly rasters, rates and
# the mean u and x of the assembly synapses (`stp_plot`).
# Runtime: about 80 s.
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie
using Statistics, StatsBase, Distributions
include("mongillo_WM2008.jl")

# Rate of an input that starts at the beginning of each interval at `peak` and decays with time constant `decay`.
function attack_decay(t, variables)
    for int in variables[:intervals]
        if int[1] <= t <= int[2]
            return variables[:peak] * exp(-(t - int[1]) / variables[:decay])
        end
    end
    return 0.0f0
end

"""
    stp_plot(model, interval, assemblies, stimuli = nothing)

Raster of the neurons of the first assembly, firing rate of every assembly, and the mean u and x of the
presynaptic neurons of the first assembly. `stimuli` is a list of [start, stop] windows shown as shaded bands.
"""
function stp_plot(model, interval, assemblies, stimuli = nothing)
    fig = Figure(size = (800, 800))
    ax1 = Axis(fig[1, 1], ylabel = "Neuron (assembly 1)")
    spikes = SNN.spiketimes(model.pop.E)
    for (row, n) in enumerate(assemblies[1].neurons[1:100])
        st = spikes[n]
        st = st[(st .>= first(interval)) .& (st .<= last(interval))]
        scatter!(ax1, st ./ 1000, fill(row, length(st)), color = :black, markersize = 2)
    end
    ax2 = Axis(fig[2, 1], ylabel = "Rate (Hz)")
    fr, r = SNN.firing_rate(model.pop.E; interval, interpolate = false)   # neurons x time
    for a in assemblies
        lines!(ax2, r ./ 1000, vec(mean(fr[a.neurons, :], dims = 1)), label = string(a.name))
    end
    axislegend(ax2, position = :rt)
    ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "u, x")
    x = SNN.record(model.syn.EE, :STPVars_x)
    u = SNN.record(model.syn.EE, :STPVars_u)
    idx = assemblies[1].neurons     # u and x are variables of the presynaptic neurons
    r = interval
    lines!(ax3, r ./ 1000, vec(mean(hcat([x(i, r) for i in idx[1:50]]...), dims = 2)), color = :red, label = "x")
    lines!(ax3, r ./ 1000, vec(mean(hcat([u(i, r) for i in idx[1:50]]...), dims = 2)), color = :blue, label = "u")
    axislegend(ax3, position = :rt)
    if stimuli !== nothing
        for ax in (ax1, ax2, ax3), s in stimuli
            vspan!(ax, s[1] / 1000, s[2] / 1000, color = (:orange, 0.2))
        end
    end
    for ax in (ax1, ax2, ax3)
        xlims!(ax, first(interval) / 1000, last(interval) / 1000)
    end
    return fig
end

##
model, assemblies = Mongillo2008(n_assemblies = 2)

peak_rate = 2kHz
stim_parameters = Dict{Symbol,Any}(:decay => 1ms, :peak => peak_rate, :start => peak_rate)
intervals = [[-1, -1], [-1, -1]]          # no cue in this run
stim_assembly = Dict(
    assembly.name => begin
        variables = merge(stim_parameters, Dict{Symbol,Any}(:intervals => [intervals[assembly.id]]))
        param = SNN.PoissonVariable(variables = variables, rate = attack_decay, μ = 1pF)
        SNN.PoissonStimulus(model.pop.E, :ge; neurons = assembly.neurons, param = param, name = string(assembly.name))
    end for assembly in assemblies
)

model = SNN.compose(model, stim_assembly)
SNN.monitor!([model.pop...], [:fire], sr = 50Hz)
SNN.monitor!(model.pop.E, [:v], sr = 50Hz)
SNN.monitor!(model.syn.EE, [:u, :x], variables = :STPVars, sr = 50Hz)

## Training
SNN.train!(model = model, duration = 1.3s, dt = 0.125f0)

root = datadir("working_memory", "Mongillo2008")
mkpath(root)
info = (neurons = 8000, assemblies = 2)
SNN.save_model(; model, path = root, name = "8000_neurons_oneitem_nostim", info, config = (; assemblies))
interval = 0s:10ms:SNN.get_time(model)

stp_plot(model, interval, assemblies)

## Reload, strengthen the assemblies and reactivate them with 100 ms current pulses
loaded = SNN.load_model(root, "8000_neurons_oneitem_nostim", info)
model, assemblies = loaded.model, loaded.config.assemblies
SNN.train!(model = model, duration = 0.5s, dt = 0.125f0)
μee_assembly = 0.48 * 8000 / model.pop.E.N * 2.5
SNN.update_weights!(model.syn.EE, assemblies[1].neurons, assemblies[1].neurons, μee_assembly)
SNN.update_weights!(model.syn.EE, assemblies[2].neurons, assemblies[2].neurons, μee_assembly)

input = 0.08pA
stimuli = []
for x = 1:15
    my_stim_int = [SNN.get_time(model)]
    model.stim.E.param.I_base .= input
    SNN.train!(model = model, duration = 0.10s, dt = 0.125f0)
    push!(my_stim_int, SNN.get_time(model))
    model.stim.E.param.I_base .= 0.0f0
    push!(stimuli, my_stim_int)
    SNN.train!(model = model, duration = 0.20s, dt = 0.125f0)
end
interval = 1.8s:10ms:SNN.get_time(model)
stp_plot(model, interval, assemblies, stimuli)
