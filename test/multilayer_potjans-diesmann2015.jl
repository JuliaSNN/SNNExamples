using Revise
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
using SNNPlots
#gr()
using Statistics, Random
using StatsBase

using Plots
#using JLD
using ProgressMeter

using SpikingNeuralNetworks
using SNNUtils
using JLD2
using Distributions
using Test

using DataFrames
using Plots

using Random

function potjans_neurons(scale=1.0)
    ccu = Dict(
        :E23 => trunc(Int32, 20683 * scale), 
        :E4 => trunc(Int32, 21915 * scale),
        :E5 => trunc(Int32, 4850 * scale),  
        :E6 => trunc(Int32, 14395 * scale),
        :I6 => trunc(Int32, 2948 * scale),  
        :I23 => trunc(Int32, 5834 * scale),
        :I5 => trunc(Int32, 1065 * scale),  
        :I4 => trunc(Int32, 5479 * scale),
        :Th => trunc(Int32, 902 * scale)
    )
    #potjans2015_param = SNN.IFCurrentParameter(; El = -49mV,τm=10ms,τabs=2ms,Vt=−50mV,Vr=-65mV)

    neurons = Dict{Symbol, SNN.AbstractPopulation}()
    for (k, neuron_count) in pairs(ccu)
        
        neurons[k] = SNN.IF(; N = neuron_count, param = SNN.IFParameter(τm = 20ms, El = -50mV), name=string(k))
        #IFCurrent(N = v, param=potjans2015_param, name=string(k))
        
    end
    return neurons
end

"""
Define Potjans parameters for neuron populations and connection probabilities
# adding static synaptic weight parameters 
# based on PSPs that are more accurate to PD based on the PyNN
# model
# Descrepency between PyNN model and original paper
# original paper
# w ± δw 	87.8 ± 8.8 pA 	Excitatory synaptic strengths
# PyNN model:             syn_weight_dist = Normal(0.15, 0.1)

# g 	–4 	Relative inhibitory synaptic strength
"""
function potjans_conn(Ne)

    function j_from_name(pre, post)
        if occursin("E", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(87.8,  8.8) # Unit is pA
            return rand(syn_weight_dist)*10^-6#pA

        elseif occursin("I", String(pre)) && occursin("E", String(post))
            #syn_weight_dist = Normal(0.15, 0.1)
            return -4.0#mV
        elseif occursin("E", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(87.8,  8.8)*10^-6# # Unit is pA
            return rand(syn_weight_dist)mV

        elseif occursin("I", String(pre)) && occursin("I", String(post))
            #syn_weight_dist = Normal(0.15, 0.1)
            return -4.0#mV
        elseif occursin("Th", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(87.8,  8.8)*10^-6# # Unit is pA
            return rand(syn_weight_dist)mV

        elseif occursin("Th", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(87.8,  8.8)*10^-6# # Unit is pA
            return rand(syn_weight_dist)mV


        else 
            throw(ArgumentError("Invalid pre-post combination: $pre-$post"))
        end
    end
    pre_layer_names = [:E23, :I23, :E4, :I4, :E5, :I5, :E6, :I6, :Th]
    post_layer_names = [:E23, :I23, :E4, :I4, :E5, :I5, :E6, :I6]

    conn_probs = Float32[
        0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.0   0.0
        0.1346  0.1371 0.0316 0.0515 0.0755 0.0     0.0042 0.0   0.0 
        0.0077  0.0059 0.0497 0.135  0.0067 0.0003  0.0453 0.0   0.0983
        0.0691  0.0029 0.0794 0.1597 0.0033 0.0     0.1057 0.0   0.0619
        0.1004  0.0622 0.0505 0.0057 0.0831 0.3726  0.0204 0.0   0.0
        0.0548  0.0269 0.0257 0.0022 0.06   0.3158  0.0086 0.0   0.0 
        0.0156  0.0066 0.0211 0.0166 0.0572 0.0197  0.0396 0.225 0.0512
        0.0364  0.001  0.0034 0.0005 0.0277 0.008   0.0658 0.144 0.0196
    ]


    conn_j = zeros(Float32, size(conn_probs))
    for pre in eachindex(pre_layer_names)
        for post in eachindex(post_layer_names)
            
            conn_j[post, pre] = j_from_name(pre_layer_names[pre], post_layer_names[post])     

        end
    end

    return pre_layer_names,post_layer_names, conn_probs, conn_j
end


"""
Main function to setup Potjans layer with memory-optimized connectivity
"""
function potjans_layer(scale)

    ## Create the neuron populations
    neurons = potjans_neurons(scale)
    exc_pop = filter(x -> occursin("E", String(x)), keys(neurons))
    inh_pop = filter(x -> occursin("I", String(x)), keys(neurons))
    Ne = trunc(Int32, sum([neurons[k].N for k in exc_pop]))
    Ni = trunc(Int32, sum([neurons[k].N for k in inh_pop]))
    pre_layer_names,post_layer_names, conn_probs, conn_j = potjans_conn(Ne)
    ##
    # Added synaptic delays from distribution informed by PyNN/PD model
    ##
    # de ± δde 	1.5 ± 0.75 ms 	Excitatory synaptic transmission delays
    # di ± δdi 	0.8 ± 0.4 ms 	Inhibitory synaptic transmission delays 
    delay_dist_exc = Normal(1.5, 0.75) # Unit is ms
    delay_dist_inh = Normal(0.8, 0.4) # Unit is ms
    connections = Dict()
    conn_map_pre_density = Dict()
    conn_map_post_density = Dict()
    cnt=0
    for i in eachindex(pre_layer_names)
        for j in eachindex(post_layer_names)
            pre = pre_layer_names[i]
            post = post_layer_names[j]
            p = conn_probs[j, i]
            @show(p)
            J = conn_j[j, i]
            sym = J>=0 ? :ge : :gi
            μ = J
            #@show(μ)
            if J>=0      
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ, p=p, σ=0, delay_dist=delay_dist_exc)
            
            else
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ, p=p, σ=0, delay_dist=delay_dist_inh)
            end
            keyed = Symbol(string(pre,post))
            if haskey(conn_map_pre_density, keyed)
                conn_map_pre_density[keyed] += length(s.I)
                conn_map_post_density[keyed] += length(s.J)

            else
                conn_map_pre_density[keyed] = length(s.I)  # or set to some default value
                conn_map_post_density[keyed] = length(s.J)
            end
            conn_map_pre_density[Symbol(string(pre,post))] += length(s.I)
            connections[Symbol(string(i,"_",pre,"_",j,"_",post))] = s
        end
    end

    # Name 	L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 	Th 	
    # Population size, N 	20 683 	5834 	21 915 	5479 	4850 	1065 	14 395 	2948 	902 	
    # External inputs, kext (reference) 	1600 	1500 	2100 	1900 	2000 	1900 	2900 	2100 	n/a 	
    # External inputs, kext (layer independent) 	2000 	1850 	2000 	1850 	2000 	1850 	2000 	1850 	n/a 	
    # Background rate, νbg 	8 Hz
    νe = 8Hz # background stimulation
    stimuli = Dict()
    μ=0.0525
    peak_rate = 2kHz
    stim_parameters = Dict(:decay=>1ms, :peak=>peak_rate, :start=>peak_rate)
    
    param = PSParam(rate=attack_decay, 
    variables=variables)
    SNN.PoissonStimulus(model.pop.E, :ge, μ=1pF, cells=assembly.cells, param=param, name=string(assembly.name))

    SE23 = SNN.PoissonStimulus(neurons[:E23], :ge, nothing; cells=[i for i in range(1,neurons[:E23].N)],μ=μ, param = νe,N=1600)
    stimuli[Symbol(string("Poisson_SE23", :E23))] = SE23
    SI23 = SNN.PoissonStimulus(neurons[:I23], :ge, nothing; cells=[i for i in range(1,neurons[:I23].N)],μ=μ, param = νe,N=1500)
    stimuli[Symbol(string("Poisson_SI23", :I23))] = SI23
    SE4 = SNN.PoissonStimulus(neurons[:E4], :ge, nothing; cells=[i for i in range(1,neurons[:E4].N)],μ=μ, param = νe,N=2100)
    stimuli[Symbol(string("Poisson_SE4", :E4))] = SE4
    SI4 = SNN.PoissonStimulus(neurons[:I4], :ge, nothing; cells=[i for i in range(1,neurons[:I4].N)],μ=μ, param = νe,N=1900)
    stimuli[Symbol(string("Poisson_SI4", :I4))] = SI4
    SE5 = SNN.PoissonStimulus(neurons[:E5], :ge, nothing; cells=[i for i in range(1,neurons[:E5].N)],μ=μ, param = νe,N=2000)
    stimuli[Symbol(string("Poisson_SE5", :E5))] = SE5
    SI5 = SNN.PoissonStimulus(neurons[:I5], :ge, nothing; cells=[i for i in range(1,neurons[:I5].N)],μ=μ, param = νe,N=1900)
    stimuli[Symbol(string("Poisson_SI5", :I5))] = SI5
    SE6 = SNN.PoissonStimulus(neurons[:E6], :ge, nothing; cells=[i for i in range(1,neurons[:E6].N)],μ=μ,  param = νe,N=2900,name=string("SE6"))
    stimuli[Symbol(string("Poisson_SE6", SE6))] = SE6
    SI6 = SNN.PoissonStimulus(neurons[:I6], :ge, nothing; cells=[i for i in range(1,neurons[:I6].N)],μ=μ,param = νe,N=2100,name=string("I6"))
    stimuli[Symbol(string("Poisson_SI6", :I6))] = SI6
    for (ind,pop) in enumerate(exc_pop)
        #νe = 4Hz # background stimulation
        post = neurons[pop]
        #@show(μ=1/100)
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=μ, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    for (ind,pop) in enumerate(inh_pop)
        #νe = 4Hz # background stimulation
        post = neurons[pop]
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=μ, name="PoissonI_$(post.name)")
        stimuli[Symbol(string("PoissonI_", pop))] = s
    end
    
    return merge_models(neurons,connections, stimuli),neurons,connections,stimuli,conn_map_pre_density,conn_map_post_density,pre_layer_names,post_layer_names

end

if !isfile("data_connectome.jld")
    model,neurons,connections,stimuli,conn_map_pre_density,conn_map_post_density,pre_layer_names,post_layer_names = potjans_layer(0.125)
    pre_synapses = []
    post_synapses = []
    layer_names = []

    for (k,s) in pairs(connections)
        push!(pre_synapses,s.I)
        push!(post_synapses,s.J)
        push!(layer_names,k)

    end
    @save "data_connectome.jld" pre_synapses post_synapses layer_names conn_map_pre_density conn_map_post_density pre_layer_names post_layer_names neurons 
    #@save "just_model.jld" model

else
    @load "data_connectome.jld" pre_synapses post_synapses layer_names conn_map_pre_density conn_map_post_density pre_layer_names post_layer_names neurons 
    #@load "just_model.jld" model

end

model,neurons,connections,stimuli,conn_map_pre_density,conn_map_post_density,pre_layer_names,post_layer_names = potjans_layer(0.125)
@save "data_connectome.jld" pre_synapses post_synapses layer_names conn_map_pre_density conn_map_post_density pre_layer_names post_layer_names neurons 
#@save "just_model.jld" model

function pre_process_for_sankey(pre_layer_names,post_layer_names,conn_map_pre_density)
    labels = []
    connections = []
    for i in eachindex(pre_layer_names)
        for j in eachindex(post_layer_names)
            pre = pre_layer_names[i]
            post = post_layer_names[j]
            push!(labels,string(pre))
            push!(connections,(i,j,conn_map_pre_density[Symbol(string(pre,post))]))
        end
    end
    @save "sankey_data.jld" pre_layer_names post_layer_names connections labels
end

#include("sankey_only.jl")
#sankey_applied(true)
sankeyed=false
if sankeyed
    sources = pre_synapses[1]  # Indices of source nodes
    targets = post_synapses[1]  # Indices of target nodes
    values_ = [1 for i in range(1,length(sources))] # Values or densities for ribbons
    layer_names  = [layer_names[ind] for (ind,post) in enumerate(post_synapses) if length(post) != 0]
    pre_synapses = [pre for pre in pre_synapses if length(pre) != 0]
    post_synapses = [post for post in post_synapses if length(post) != 0]

    pre_process_for_sankey(pre_layer_names,post_layer_names,conn_map_pre_density)

    p = SNNPlots.doparallelCoords_optimized(pre_synapses,post_synapses;figure_name="parallelCoordinatesPlot.png")
end
    #p = SNNPlots.doparallelCoords(pre_synapses,post_synapses)
#Plots.plot(p)
#Plots.savefig("parallelCoordinatesPlot.png")


#@snn_kw struct PoissonStimulusInterval{R=Float32, } <: PoissonStimulusParameter
#    rate::Vector{R} = fill(0.0, N)
#    intervals::Vector{Vector{R}}
#end

param = PoissonStimulusInterval(rate,[[]])
TotalNNeurons = sum([v.N for v in values(neurons)])
duration = 1500ms
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [:v], sr=400Hz)
SNN.sim!(model=model; duration = duration, pbar = true, dt = 0.125)
gui(SNN.raster(model.pop, [10s, 15s]))
gui(SNN.raster(model.pop, [14.5s, 15s]))
gui(SNN.raster(model.pop.E23, [14.5s, 15s]))

#gui(plot(interval, mean.(frE), label=hcat(names_pop...), xlabel="Time [ms]", ylabel="Firing rate [Hz]", legend=:topleft))
gui(vecplot(model.pop.E23, :v, neurons =1:200, r=0s:0.0125s,label="soma"))
gui(vecplot(model.pop.Th, :v, neurons =1:50, r=0s:0.125s,label="soma"))
gui(vecplot(model.pop.I4, :v, neurons =1:200, r=0s:0.125s,label="soma"))
gui(vecplot(model.pop.E4, :v, neurons =1:200, r=0s:0.125s,label="soma"))
#gui(vecplot(model.pop.E23, :v, neurons =1:200, r=0s:0.0125s,label="soma"))
layer_names, conn_probs, conn_j = potjans_conn(4000)
pj = heatmap(conn_j, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:bluesreds,  title="Synaptic weights", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500), clims=(-maximum(abs.(conn_j)), maximum(abs.(conn_j))))
gui(pj)
Trange = 0s:0.25:1.5s
frE, interval, names_pop = SNN.firing_rate(model.pop, interval = Trange)
gui(scatter([i for i in range(1,9)],mean.(frE), labels=names_pop, xlabel="Time [ms]", ylabel="Firing rate [Hz]", legend=:topright))
