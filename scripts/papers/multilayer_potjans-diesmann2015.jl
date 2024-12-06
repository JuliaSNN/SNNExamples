using Revise
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter, IFParameter
using Statistics, Random
using Plots
using SparseArrays
using ProgressMeter
using Plots
using SpikingNeuralNetworks
using SNNUtils
using JLD2
using Distributions

"""
Auxiliary Potjans parameters for neural populations with scaled cell counts
"""
function potjans_neurons(scale=1.0)
    ccu = Dict(
        :E23 => trunc(Int32, 20683 * scale), 
        :E4 => trunc(Int32, 21915 * scale),
        :E5 => trunc(Int32, 4850 * scale),  
        :E6 => trunc(Int32, 14395 * scale),
        :I6 => trunc(Int32, 2948 * scale),  
        :I23 => trunc(Int32, 5834 * scale),
        :I5 => trunc(Int32, 1065 * scale),  
        :I4 => trunc(Int32, 5479 * scale)
    )

    neurons = Dict{Symbol, SNN.AbstractPopulation}()
    for (k, v) in ccu
        if occursin("E", String(k))
	    # PD model only consists of IF neurons.
            neurons[k] = IF(N = v, param=LKD2014SingleExp.PV, name=string(k))
        else
            neurons[k] = IF(N = v, param=LKD2014SingleExp.PV, name=string(k))
        end
    end
    return neurons
end

"""
Define Potjans parameters for neuron populations and connection probabilities
"""
function potjans_conn(Ne)
    # adding static synaptic weight parameters 
    # based on PSPs that are more accurate to PD based on the PyNN
    # model
    function j_from_name(pre, post)
        if occursin("E", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            
            return rand(syn_weight_dist)
        elseif occursin("I", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            return -4.0*rand(syn_weight_dist)

        elseif occursin("E", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            
            return rand(syn_weight_dist)
        elseif occursin("I", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            return -4.0*rand(syn_weight_dist)
        else 
            throw(ArgumentError("Invalid pre-post combination: $pre-$post"))
        end
    end

    

    layer_names = [:E23, :I23, :E4, :I4, :E5, :I5, :E6, :I6]


    # Replace static matrix with a regular matrix for `conn_probs`
    # ! the convention is j_post_pre. This is how the matrices `w` are built. Are you using that when defining the parameters?
    conn_probs = Float32[
        0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.0    
        0.1346  0.1371 0.0316 0.0515 0.0755 0.0     0.0042 0.0    
        0.0077  0.0059 0.0497 0.135  0.0067 0.0003  0.0453 0.0    
        0.0691  0.0029 0.0794 0.1597 0.0033 0.0     0.1057 0.0    
        0.1004  0.0622 0.0505 0.0057 0.0831 0.3726  0.0204 0.0    
        0.0548  0.0269 0.0257 0.0022 0.06   0.3158  0.0086 0.0    
        0.0156  0.0066 0.0211 0.0166 0.0572 0.0197  0.0396 0.2252
        0.0364  0.001  0.0034 0.0005 0.0277 0.008   0.0658 0.1443
    ]


    conn_j = zeros(Float32, size(conn_probs))
    for pre in eachindex(layer_names)
        for post in eachindex(layer_names)
            
            conn_j[post, pre ] = j_from_name(layer_names[pre], layer_names[post])     

        end
    end

    return layer_names, conn_probs, conn_j
end

function unused_volumetric_part_of_model()

    total_cortical_thickness = 1500.0
    N_full = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948] 
    N_E_total = N_full[1]+N_full[3]+N_full[5]+N_full[7]
    dimensions_3D = Dict(
        "x_dimension"=> 1000,
        "z_dimension"=>  1000,
        "total_cortical_thickness"=> total_cortical_thickness,

        # Have the thicknesses proportional to the numbers of E cells in each layer
        "layer_thicknesses"=> Dict(
        "L23"=> total_cortical_thickness*N_full[1]/N_E_total,
        "L4" => total_cortical_thickness*N_full[3]/N_E_total,
        "L5" => total_cortical_thickness*N_full[5]/N_E_total,
        "L6" => total_cortical_thickness*N_full[7]/N_E_total,
        "thalamus" => 100
        )
    )
    
    net_dict = Dict{String, Any}(
        "PSP_e"=> 0.15,
        # Relative standard deviation of the postsynaptic potential.
        "PSP_sd"=> 0.1,
        # Relative inhibitory synaptic strength (in relative units).
        "g"=> -4,
        # Rate of the Poissonian spike generator (in Hz).
        "bg_rate"=> 8.,
        # Turn Poisson input on or off (True or False).
        "poisson_input"=> true,
        # Delay of the Poisson generator (in ms).
        "poisson_delay"=> 1.5,
        # Mean delay of excitatory connections (in ms).
        "mean_delay_exc"=> 1.5,
        # Mean delay of inhibitory connections (in ms).
        "mean_delay_inh"=> 0.75,
        # Relative standard deviation of the delay of excitatory and
        # inhibitory connections (in relative units).
        "rel_std_delay"=> 0.5
    )
    dimensions_3D,net_dict


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
    layer_names, conn_probs, conn_j = potjans_conn(Ne)
    ##
    # Added synaptic delays from distribution informed by PyNN/PD model
    ##
    syn_weight_dist = Normal(0.15, 0.1)
    delay_dist_exc = Normal(1.5, 0.5)
    delay_dist_inh = Normal( 0.75, 0.5)

    ## Create the synaptic connections based on the connection probabilities and synaptic weights assigned to each pre-post pair
    connections = Dict()
    for i in eachindex(layer_names)
        for j in eachindex(layer_names)
            pre = layer_names[i]
            post = layer_names[j]
            p = conn_probs[j, i]
            J = conn_j[j, i]
            sym = J>=0 ? :ge : :gi
            μ = abs(J)
            if J>=0        
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ, p=p, σ=0, delay_dist=delay_dist_exc)
            else
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = -μ, p=p, σ=0, delay_dist=delay_dist_inh)
            
            end
            connections[Symbol(string(pre,"_", post))] = s
        end
    end

    ## Create the Poisson stimulus for each population
    full_mean_rates = [0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523]
    stimuli = Dict()
    for (ind,pop) in enumerate(exc_pop)
        νe = full_mean_rates[ind]kHz
        post = neurons[pop]
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1.f0, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    return merge_models(neurons,connections, stimuli),neurons,connections,stimuli
end


model,neurons,connections,stimuli = potjans_layer(0.01)
duration = 15000ms
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [:v], sr=200Hz)
SNN.sim!(model=model; duration = duration, pbar = true, dt = 0.125)
SNN.raster(model.pop, [10s, 15s])

Trange = 5s:10:15s
frE, interval, names_pop = SNN.firing_rate(model.pop, interval = Trange)
plot(interval, mean.(frE), label=hcat(names_pop...), xlabel="Time [ms]", ylabel="Firing rate [Hz]", legend=:topleft)
##

vecplot(model.pop.E23, :v, neurons =1, r=0s:15s,label="soma")
layer_names, conn_probs, conn_j = potjans_conn(4000)
pj = heatmap(conn_j, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:bluesreds,  title="Synaptic weights", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500), clims=(-maximum(abs.(conn_j)), maximum(abs.(conn_j))))
pprob=heatmap(conn_probs, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:viridis,  title="Connection probability", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500))
plot(pprob, pj, layout=(1,2), size=(1000,500), margin=5Plots.mm)
##
