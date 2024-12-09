using Revise
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter, IFParameter
using Statistics, Random
using Plots
#using SparseArrays
#using ProgressMeter
using Plots
using SpikingNeuralNetworks
using SNNUtils
using JLD2
using Distributions
include("Tonic_NMNIST_Stimulus.jl")
using .Tonic_NMNIST_Stimulus

"""
Model parameters pasted here from original 2015 paper.

https://pmc.ncbi.nlm.nih.gov/articles/PMC3920768/

Layer-specific external inputs
External inputs 	L2/3 	L4 	    L5 	    L6
Total 	            1606 	2111 	1997 	2915


Parameter specification
Populations and inputs
Name 	L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 	Th 	
Population size, N 	20 683 	5834 	21 915 	5479 	4850 	1065 	14 395 	2948 	902 	
External inputs, kext (reference) 	1600 	1500 	2100 	1900 	2000 	1900 	2900 	2100 	n/a 	
External inputs, kext (layer independent) 	2000 	1850 	2000 	1850 	2000 	1850 	2000 	1850 	n/a 	
Background rate, νbg 	8 Hz
Connectivity
		
        from (columns are presynaptic, rows are post synaptic) 				
		L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 	Th
to 	L2/3e 	0.101 	0.169 	0.044 	0.082 	0.032 	0.0 	0.008 	0.0 	0.0
	L2/3i 	0.135 	0.137 	0.032 	0.052 	0.075 	0.0 	0.004 	0.0 	0.0
	L4e 	0.008 	0.006 	0.050 	0.135 	0.007 	0.0003 	0.045 	0.0 	0.0983
	L4i 	0.069 	0.003 	0.079 	0.160 	0.003 	0.0 	0.106 	0.0 	0.0619
	L5e 	0.100 	0.062 	0.051 	0.006 	0.083 	0.373 	0.020 	0.0 	0.0
	L5i 	0.055 	0.027 	0.026 	0.002 	0.060 	0.316 	0.009 	0.0 	0.0
	L6e 	0.016 	0.007 	0.021 	0.017 	0.057 	0.020 	0.040 	0.225 	0.0512
	L6i 	0.036 	0.001 	0.003 	0.001 	0.028 	0.008 	0.066 	0.144 	0.0196
Name 	Value 	Description
w ± δw 	87.8 ± 8.8 pA 	Excitatory synaptic strengths
g 	–4 	Relative inhibitory synaptic strength
de ± δde 	1.5 ± 0.75 ms 	Excitatory synaptic transmission delays
di ± δdi 	0.8 ± 0.4 ms 	Inhibitory synaptic transmission delays
Neuron model
Name 	Value 	Description

τm 	10 ms 	Membrane time constant
τref 	2 ms 	Absolute refractory period
τsyn 	0.5 ms 	Postsynaptic current time constant
Cm 	250 pF 	Membrane capacity
Vreset 	−65 mV 	Reset potential
θ 	−50 mV 	Fixed firing threshold
νth 	15 Hz 	Thalamic firing rate during input perio


Populations 	Nine; 8 cortical populations and 1 thalamic population
Topology 	—
Connectivity 	Random connections
Neuron model 	Cortex: Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory period (voltage clamp), thalamus: Fixed-rate Poisson
Synapse model 	Exponential-shaped postsynaptic currents
Plasticity 	—
Input 	Cortex: Independent fixed-rate Poisson spike trains
Measurements 	Spike activity, membrane potentials
Populations
Type 	Elements
Cortical network 	iaf neurons, 8 populations (2 per layer), type specific size N
Th 	Poisson, 1 population, size Nth
Connectivity
Type 	Random connections with independently chosen pre- and postsynaptic neurons; see Table 5 for probabilities
Weights 	Fixed, drawn from Gaussian distribution
Delays 	Fixed, drawn from Gaussian distribution multiples of computation stepsize
Neuron and synapse model
Name 	iaf neuron
Type 	Leaky integrate-and-fire, exponential-shaped synaptic current inputs
Subthreshold dynamics 	Inline graphic
	V(t) = Vreset else
	Inline graphic
Spiking 	If Inline graphic
	1. set t* = t, 2. emit spike with time stamp t*
Input
Type 	Description
Background 	Independent Poisson spikes to iaf neurons (Table 5)
Measurements
Spiking activity and membrane potentials from a subset of neurons in every population
"""


"""
Auxiliary Potjans parameters for neural populations with scaled cell counts
Name 	L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 	Th 	
Population size, N 	20683 	5834 	21 915 	5479 	4850 	1065 	14395 	2948 	902 	

"""

"""
τm 	10 ms 	Membrane time constant
τref 	2 ms 	Absolute refractory period
τsyn 	0.5 ms 	Postsynaptic current time constant
Cm 	250 pF 	Membrane capacity
Vreset 	−65 mV 	Reset potential
θ 	−50 mV 	Fixed firing threshold
νth 	15 Hz 	Thalamic firing rate during input period
  
τm::FT = 20ms # Membrane time constant
Vt::FT = -50mV # Membrane potential threshold
Vr::FT = -60mV # Reset potential
El::FT = -70mV # Resting membrane potential
R::FT = nS / gL # 40nS Membrane conductance
ΔT::FT = 2mV # Slope factor
τabs::FT = 2ms # Absolute refractory period
#synapses
τe::FT = 6ms # Rise time for excitatory synapses
τi::FT = 2ms # Rise time for inhibitory synapses
E_i::FT = -75mV # Reversal potential
E_e::FT = 0mV # Reversal potential

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
        :I4 => trunc(Int32, 5479 * scale),
        :Th => trunc(Int32, 902 * scale)
    )
    potjans2015_param = SNN.IFCurrentParameter(; El = -49mV,τm=10ms,τabs=2ms,Vt=−50mV,Vr=-65mV)

    neurons = Dict{Symbol, SNN.AbstractPopulation}()
    for (k, v) in ccu
        
        neurons[k] = IFCurrent(N = v, param=potjans2015_param, name=string(k))
        
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
    # Descrepency between PyNN model and original paper
    # original paper
    # w ± δw 	87.8 ± 8.8 pA 	Excitatory synaptic strengths
    # PyNN model:             syn_weight_dist = Normal(0.15, 0.1)

    # g 	–4 	Relative inhibitory synaptic strength
    #
    # 
    function j_from_name(pre, post)
        if occursin("E", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(87.8,  8.8) # Unit is pA
            return rand(syn_weight_dist)pA

        elseif occursin("I", String(pre)) && occursin("E", String(post))
            #syn_weight_dist = Normal(0.15, 0.1)
            return -4.0pA
        elseif occursin("E", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(87.8,  8.8) # Unit is pA
            return rand(syn_weight_dist)pA

        elseif occursin("I", String(pre)) && occursin("I", String(post))
            #syn_weight_dist = Normal(0.15, 0.1)
            return -4.0pA
        elseif occursin("Th", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(87.8,  8.8) # Unit is pA
            return rand(syn_weight_dist)pA

        elseif occursin("Th", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(87.8,  8.8) # Unit is pA
            return rand(syn_weight_dist)pA


        else 
            throw(ArgumentError("Invalid pre-post combination: $pre-$post"))
        end
    end

    

    pre_layer_names = [:E23, :I23, :E4, :I4, :E5, :I5, :E6, :I6, :Th]
    post_layer_names = [:E23, :I23, :E4, :I4, :E5, :I5, :E6, :I6]


    # Replace static matrix with a regular matrix for `conn_probs`
    # ! the convention is j_post_pre. This is how the matrices `w` are built. Are you using that when defining the parameters?
    # 		L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 	Th

    """
            from (columns are presynaptic, rows are post synaptic) 				
		L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 	Th
to 	L2/3e 	0.101 	0.169 	0.044 	0.082 	0.032 	0.0 	0.008 	0.0 	0.0
	L2/3i 	0.135 	0.137 	0.032 	0.052 	0.075 	0.0 	0.004 	0.0 	0.0
	L4e 	0.008 	0.006 	0.050 	0.135 	0.007 	0.0003 	0.045 	0.0 	0.0983
	L4i 	0.069 	0.003 	0.079 	0.160 	0.003 	0.0 	0.106 	0.0 	0.0619
	L5e 	0.100 	0.062 	0.051 	0.006 	0.083 	0.373 	0.020 	0.0 	0.0
	L5i 	0.055 	0.027 	0.026 	0.002 	0.060 	0.316 	0.009 	0.0 	0.0
	L6e 	0.016 	0.007 	0.021 	0.017 	0.057 	0.020 	0.040 	0.225 	0.0512
	L6i 	0.036 	0.001 	0.003 	0.001 	0.028 	0.008 	0.066 	0.144 	0.0196

    """




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

    ## Create the synaptic connections based on the connection probabilities and synaptic weights assigned to each pre-post pair
    connections = Dict()
    stdp_param = STDPParameter(A_pre =5e-1, 
        A_post=-5e-1,
        τpre  =20ms,
        τpost =15ms)

    for i in eachindex(pre_layer_names)
        for j in eachindex(post_layer_names)
            pre = pre_layer_names[i]
            post = post_layer_names[j]
            p = conn_probs[j, i]
            J = conn_j[j, i]
            sym = J>=0 ? :ge : :gi
            μ = abs(J)
            if J>=0        
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ, p=p, σ=0,param=stdp_param, delay_dist=delay_dist_exc)
            else
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = -μ, p=p, σ=0, delay_dist=delay_dist_inh)
            
            end
            connections[Symbol(string(pre,"_", post))] = s
        end
    end

    ## Create the Poisson stimulus for each population
    ## The order of these rates needs to be carefully checked as I have changed the order of populations from the 
    # PyNN model to conveniently group excitatory and inhibitory connections. 
    #full_mean_rates = [0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523]
    stimuli = Dict()
    for (ind,pop) in enumerate(exc_pop)
        νe = 8Hz # background stimulation
        #full_mean_rates[ind]kHz
        post = neurons[pop]
        ##
        # TODO replace all, with something that better matches the number of poisson spike generator sources and targets from the Potjan's model.
        ##
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1.f0, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    for (ind,pop) in enumerate(inh_pop)
        νe = 8Hz # background stimulation
        #full_mean_rates[ind]kHz
        post = neurons[pop]
        ##
        # TODO replace all, with something that better matches the number of poisson spike generator sources and targets from the Potjan's model.
        ##

        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1.f0, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    
    return merge_models(neurons,connections, stimuli),neurons,connections,stimuli
end

#if !isfile("cached_potjans_model.jld2")
    model,neurons_,connections_,stimuli_,net_dict = potjans_layer(0.085)
    @save "cached_potjans_model.jld2" model neurons_ connections_ stimuli_ net_dict
#else
    @load "cached_potjans_model.jld2" model neurons_ connections_ stimuli_ net_dict
#end
#@show(connections_.vals)

before_learnning_weights = model.syn[1].W
#=
ΔTs = -100:1:100ms
ΔWs = zeros(Float32, length(ΔTs))
Threads.@threads for i in eachindex(ΔTs)
    ΔT = ΔTs[i]
    #spiketime = [2000ms, 2000ms+ΔT]
    #neurons = [[1], [2]]
    #inputs = SpikeTime(spiketime, neurons)
    w = zeros(Float32, 2,2)
    w[1, 2] = 1f0
    st = Identity(N=max_neurons(inputs))
    stim = SpikeTimeStimulusIdentity(st, :g, param=inputs)
    syn = SpikingSynapse( st, st, nothing, w = w,  param = stdp_param)
    
    model = merge_models(pop=st, stim=stim, syn=syn, silent=true)
    SNN.monitor(model.pop..., [:fire])
    SNN.monitor(model.syn..., [:tpre, :tpost])
    train!(model=model, duration=3000ms, dt=0.1ms)
    ΔWs[i] = model.syn[1].W[1] - 1
end
=#

duration = 15000ms
SNN.monitor([model.pop...], [:fire])
#SNN.monitor([model.pop...], [:v], sr=200Hz)
SNN.sim!(model=model, duration=3000ms, dt=0.125, pbar = true)#, dt = 0.125)

#=
# Example data: Spike times and trial start times
s#pike_times = [0.1, 0.15, 0.2, 0.4, 0.6, 0.65, 0.8, 1.0, 1.1, 1.3, 1.5]
t#rial_starts = [0.0, 1.0, 2.0]  # Start times of each trial
num_trials = length(trial_starts)

# Parameters
bin_width = 0.1  # Bin width in seconds
time_window = (0.0, 1.0)  # Time window relative to each trial start

# Create bins
edges = collect(time_window[1]:bin_width:time_window[2])
num_bins = length(edges) - 1

# Initialize a matrix to hold spike counts
spike_counts = zeros(num_trials, num_bins)

# Bin spikes for each trial
for (i, trial_start) in enumerate(trial_starts)
    aligned_spikes = spike_times .- trial_start  # Align spikes to trial start
    filtered_spikes = aligned_spikes[aligned_spikes .>= time_window[1] .& aligned_spikes .< time_window[2]]
    spike_counts[i, :] = histcounts(filtered_spikes, edges)
end

# Compute average spike activity (optional)
average_spike_activity = mean(spike_counts, dims=1)

# Plot heatmap
heatmap(
    edges[1:end-1] .+ bin_width / 2,  # Bin centers
    1:num_trials,  # Trial indices
    spike_counts,
    xlabel="Time (s)",
    ylabel="Trial",
    color=:viridis,
    title="Spike Activity Heatmap"
)
=#
SNN.train!(model=model, duration=3000ms, dt=0.125, pbar = true)#, dt = 0.125)

#SNN.sim!(model=model; duration = duration, pbar = true, dt = 0.125)
display(SNN.raster(model.pop, [1.0s, 2s]))
savefig("without_stimulus.png")




after_learnning_weights0 = model.syn[1].W

training_order = shuffle(0:60000-1)
cached_spikes=[]
for i in 1:40
    push!(cached_spikes,Tonic_NMNIST_Stimulus.getNMNIST(training_order[i]))
end

"""
Filter for spike packets that belong to provided labels that pertain to single numeral characters.
"""


if !isfile("labeled_packets.jld2")
    labeled_packets = Tonic_NMNIST_Stimulus.spike_packets_by_labels(cached_spikes)
    filter_label = 0
    (population_code,time_and_offset) = labeled_packets[filter_label]
    for filter_label in range(0,9)
        (population_code_,time_and_offset_) = labeled_packets[filter_label]
        append!(population_code,population_code_)
        append!(time_and_offset,time_and_offset_)
    end
    @save "labeled_packets.jld2" population_code time_and_offset
else
    @load "labeled_packets.jld2" population_code time_and_offset
end



p1 = Plots.scatter()
scatter!(p1,
    time_and_offset,
    population_code,    
    ms = 0.5,  # Marker size
    ylabel = "Neuron Index" ,
    xlabel ="Time (ms)",
    title = "Spiking Activity with Distinct Characters", 
    legend=false
)
display(plot(p1))
savefig("stimulus.png")

neurons_as_nested_array = [ Vector{Int64}([n]) for n in population_code]
inputs = SpikeTime(time_and_offset,neurons_as_nested_array)

st = neurons_[:E4] #Identity(N=max_neurons(inputs))
w = ones(Float32,neurons_[:E4].N,max_neurons(inputs))*15


st = Identity(N=max_neurons(inputs))
stim = SpikeTimeStimulusIdentity(st, :g, param=inputs)


syn = SpikingSynapse( st, neurons_[:E4], nothing, w = w)#,  param = stdp_param)
model2 = merge_models(pop=[st,model], stim=[stim,stimuli_], syn=[syn,connections_], silent=false)

duration = 15000ms
SNN.monitor([model2.pop...], [:fire])
SNN.monitor([model2.pop...], [:v], sr=200Hz)
SNN.train!(model=model2; duration = duration, pbar = true, dt = 0.125)
#display(SNN.raster(model2.pop, [0s, 15s]))

after_learnning_weights1 = model.syn[1].W

@show(mean(before_learnning_weights))
@show(mean(after_learnning_weights0))
@show(mean(after_learnning_weights1))

#mean(model2.syn[1].W)

SNN.spiketimes(model.pop[1])

#x, y, y0 = SNN._raster(model2.pop.pop_2_E5,[1.95s, 2s]))

display(SNN.raster(model2.pop, [1.75s, 2s]))

savefig("with_stimulus.png")

Trange = 0:10:15s
frE, interval, names_pop = SNN.firing_rate(model2.pop, interval = Trange)
plot(mean.(frE), label=hcat(names_pop...), xlabel="Time [ms]", ylabel="Firing rate [Hz]", legend=:topleft)
savefig("firing_rate.png")

##

vecplot(model2.pop.E4, :v, neurons =1, r=0s:15s,label="soma")
savefig("vector_vm_plot.png")
layer_names, conn_probs, conn_j = potjans_conn(4000)
pj = heatmap(conn_j, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:bluesreds,  title="Synaptic weights", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500), clims=(-maximum(abs.(conn_j)), maximum(abs.(conn_j))))
pprob=heatmap(conn_probs, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:viridis,  title="Connection probability", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500))
plot(pprob, pj, layout=(1,2), size=(1000,500), margin=5Plots.mm)
