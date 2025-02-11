#using PlotlyJS
#plotlyjs() 
#using PlotlyJS, ElectronDisplay
using Revise
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter, IFParameter
using Statistics, Random
    using Plots

#using SparseArrays
#using ProgressMeter
#using Plots;
#gr() 

using SpikingNeuralNetworks
using SNNUtils
using JLD2
using Distributions
using Test

#Copy
using DataFrames
#using Makie
using Statistics
using Clustering
using Distances
using StatsBase
using Plots
using StatsBase  # if not installed: import Pkg; Pkg.add("StatsBase")

#using Pkg
#using MakieLayout
#using CairoMakie
#CairoMakie.activate!(type = "svg")
#using GLMakie
#using SpikingNeuralNetworks.Graphs
#using SpikingNeuralNetworks.MetaGraphs
#using GraphRecipes
#using CairoMakie, Colors, LinearAlgebra
#using SGtSNEpi
#using SparseArrays
#using BlockArrays

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



External inputs, kext (reference) 	1600 	1500 	2100 	1900 	2000 	1900 	2900 	2100 	n/a 	
External inputs, kext (layer independent) 	2000 	1850 	2000 	1850 	2000 	1850 	2000 	1850 	n/a 	
Background rate, νbg 	8 Hz



Connectivity
		
        from (columns are presynaptic, rows are post synaptic) 				
		L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 
my_graph = SNN.graph(model)
##


##
set_theme!(theme_light())
f, ax, p = graphplot(my_graph, 
                    edge_width=[0.1 for i in 1:ne(my_graph)],
                     node_size=[30 for i in 1:nv(my_graph)],
                     arrow_shift=0.90,
                     nlabels=[get_prop(my_graph, v, :name) for v in vertices(my_graph)],
                     nlabels_distance=20,
                     )



                # f, ax, p = graphplot(my_graph, n_labels=names, nlabels_fontsize=12,node_size = 30, edge_width = .1, arrow_shift=.90)
# hidedecorations!(ax)
# names = [get_prop(my_graph, v, :name) for v in vertices(my_graph)]
deregister_interaction!(ax, :rectanglezoom)
register_interaction!(ax, :nhover, NodeHoverHighlight(p))
register_interaction!(ax, :ehover, EdgeHoverHighlight(p))
register_interaction!(ax, :ndrag, NodeDrag(p))
register_interaction!(ax, :edrag, EdgeDrag(p))
	Th
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


    # Preallocate arrays for sparse matrix construction
    #row_indices = Int[]
    #col_indices = Int[]
    #values = Float64[]

    ## Create the synaptic connections based on the connection probabilities and synaptic weights assigned to each pre-post pair
    connections = Dict()
    for i in eachindex(pre_layer_names)
        for j in eachindex(post_layer_names)
            pre = pre_layer_names[i]
            post = post_layer_names[j]
            p = conn_probs[j, i]
            J = conn_j[j, i]
            sym = J>=0 ? :ge : :gi
            μ = abs(J)
            if J>=0      
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ, p=p, σ=0, delay_dist=delay_dist_exc)
                

            else
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = -μ, p=p, σ=0, delay_dist=delay_dist_inh)
            
            end
            connections[Symbol(string(i,"_",pre,"_",j,"_",post))] = s
                # Populate the sparse matrix indices and values
            #@show(neurons[pre])
            #@show(row_indices)
            #cells=
            #push!(row_indices, [i for i in range(1,neurons[pre].N)])
            #push!(col_indices, [i for i in range(1,neurons[post].N)])
            #append!(values,[μ for i in neurons[pre]])

        end

    end
    for s in values(connections)
        @show(s.W)
        @show(s.colptr)
        @show(s.rowptr)
    end

    #=
    # Suppose we have a list of smaller matrices
    smaller_matrices = [
        rand(2, 2),    # 2x2 matrix
        rand(3, 4),    # 3x4 matrix
        rand(1, 1)     # 1x1 matrix
    ]

    # Create blocks for the giant matrix
    blocks = []
    block_sizes = [(size(mat)..., mat) for mat in smaller_matrices]

    # Define the overall dimensions of the giant matrix
    total_rows = sum(size(mat, 1) for mat in smaller_matrices)
    total_cols = maximum(size(mat, 2) for mat in smaller_matrices)

    # Initialize the giant block matrix
    giant_matrix = BlockMatrix{Float64}(total_rows, total_cols)

    # Populate the giant matrix with smaller matrices
    current_row = 1
    for (rows, cols, mat) in block_sizes
        giant_matrix[current_row:current_row + rows - 1, 1:cols] .= mat
        current_row += rows
    end
    =#

    #println(giant_matrix)

    # Create the sparse matrix
    #num_neurons_pre = length(pre_layer_names)
    #num_neurons_post = length(post_layer_names)

    #sparse_matrix = sparse(row_indices, col_indices, values, num_neurons_pre, num_neurons_post)

    ## Create the Poisson stimulus for each population
    ## The order of these rates needs to be carefully checked as I have changed the order of populations from the 
    # PyNN model to conveniently group excitatory and inhibitory connections. 
    #full_mean_rates = [0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523]


    #
    # Name 	L2/3e 	L2/3i 	L4e 	L4i 	L5e 	L5i 	L6e 	L6i 	Th 	
    # Population size, N 	20 683 	5834 	21 915 	5479 	4850 	1065 	14 395 	2948 	902 	

    # External inputs, kext (reference) 	1600 	1500 	2100 	1900 	2000 	1900 	2900 	2100 	n/a 	
    # External inputs, kext (layer independent) 	2000 	1850 	2000 	1850 	2000 	1850 	2000 	1850 	n/a 	
    # Background rate, νbg 	8 Hz

    #
    #
    νe = 8Hz # background stimulation

    stimuli = Dict()
    SE23 = SNN.PoissonStimulus(neurons[:E23], :ge, nothing; cells=[i for i in range(1,neurons[:E23].N)],μ=1pF, param = νe,N=1600)
    stimuli[Symbol(string("Poisson_SE23", :E23))] = SE23
    SI23 = SNN.PoissonStimulus(neurons[:I23], :ge, nothing; cells=[i for i in range(1,neurons[:I23].N)],μ=1pF, param = νe,N=1500)
    stimuli[Symbol(string("Poisson_SI23", :I23))] = SI23

    SE4 = SNN.PoissonStimulus(neurons[:E4], :ge, nothing; cells=[i for i in range(1,neurons[:E4].N)],μ=1pF, param = νe,N=2100)
    stimuli[Symbol(string("Poisson_SE4", :E4))] = SE4


    SI4 = SNN.PoissonStimulus(neurons[:I4], :ge, nothing; cells=[i for i in range(1,neurons[:I4].N)],μ=1pF, param = νe,N=1900)
    stimuli[Symbol(string("Poisson_SI4", :I4))] = SI4


    SE5 = SNN.PoissonStimulus(neurons[:E5], :ge, nothing; cells=[i for i in range(1,neurons[:E5].N)],μ=1pF, param = νe,N=2000)
    stimuli[Symbol(string("Poisson_SE5", :E5))] = SE5



    SI5 = SNN.PoissonStimulus(neurons[:I5], :ge, nothing; cells=[i for i in range(1,neurons[:I5].N)],μ=1pF, param = νe,N=1900)
    stimuli[Symbol(string("Poisson_SI5", :I5))] = SI5


    SE6 = SNN.PoissonStimulus(neurons[:E6], :ge, nothing; cells=[i for i in range(1,neurons[:E6].N)],μ=1pF,  param = νe,N=2900)
    stimuli[Symbol(string("Poisson_SE6", SE6))] = SE6

    SI6 = SNN.PoissonStimulus(neurons[:I6], :ge, μ=1pF, nothing; cells=[i for i in range(1,neurons[:I6].N)],param = νe,N=2100,name=string("I6"))
    #SNN.PoissonStimulus(model.pop.E, :ge, μ=1pF, cells=assembly.cells, param=param, name=string(assembly.name))
    stimuli[Symbol(string("Poisson_SI6", :I6))] = SI6

    for (ind,pop) in enumerate(exc_pop)
        νe = 8Hz # background stimulation
        post = neurons[pop]
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1pF, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    for (ind,pop) in enumerate(inh_pop)
        νe = 8Hz # background stimulation
        post = neurons[pop]

        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1pF, name="PoissonI_$(post.name)")
        stimuli[Symbol(string("PoissonI_", pop))] = s
    end
    
    return merge_models(neurons,connections, stimuli),neurons,connections,stimuli
end

if !isfile("data.jld")
    model,neurons,connections,stimuli = potjans_layer(0.125)
    pre_synapses = []
    post_synapses = []
    for s in values(connections)
        push!(pre_synapses,s.I)
        push!(post_synapses,s.J)

    end
    @save pre_synapses post_synapses "data.jld"
else:
    @load pre_synapses post_synapses "data.jld"

sources = pre_synapses[1]#[0, 1, 0, 2]  # Indices of source nodes
targets = post_synapses[1]#[2, 2, 3, 3]  # Indices of target nodes
values_ = [1 for i in range(1,length(sources))] #[8, 4, 2, 8]   # Values or densities for ribbons

function sankey_applied()
    sankey_trace = sankey(
        arrangement = "snap",
        node = attr(
            label    = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"],
            color    = ["blue", "red", "green", "orange"],
            pad      = 15,
            thickness = 20,
            line     = attr(color = "black", width = 0.5)
        ),
        link = attr(
            source = sources,
            target = targets,
            value  = values_
        )
    )

    #plt = PlotlyJS.plot(sankey_trace)
    #display(plt)
    plt = plot(sankey_trace)
    ElectronDisplay.display(plt)  # opens a new Electron window
end
#=
function parallelcoordsB(data)
    # Let's say data is size (N x M)
   # N = size(data, 1)

    # 1. Correlation among columns => 6×6 matrix
    corr_matrix = cor(data)

    # 2. Convert correlation to distance
    dist_matrix = 1 .- abs.(corr_matrix)

    # 3. Hierarchical clustering on columns
    hc = hclust(dist_matrix, linkage=:ward)

    # 4. Extract ordering from the hc object:
    reordered_cols = hc.order

    # 5. Reorder data columns
    data_reordered = data[:, reordered_cols]

    # 6. Simple parallel coordinates with optional downsampling

    p = parallelcoords(data_reordered)
    return p
end
=#
using Plots

"""
    parallelcoords(data; colnames=String[])

Create a parallel coordinates plot for the given data matrix.
Each row of `data` is one observation, each column is a dimension.

# Arguments
- `data::AbstractMatrix`: size (N × M) matrix.
- `colnames`: Optional vector of length M with column names.
    """

function parallelcoords_ragged(data::Vector{<:AbstractVector}; alpha_val=0.3)
    """
    data is a vector of vectors, each possibly of different length.
    We'll plot them all on the same figure, each vs. its own index range.
    """
    p = plot(legend=false, xlabel="Index", ylabel="Value")
    for v in data
        @show(1:length(v))
        @show(v)
        plot!(p, 1:length(v), v, alpha=alpha_val)
    end
    return p
end


function parallelcoords(data::AbstractMatrix; maxlines=200, alpha_val=0.3)
    N, M = size(data)
    #rows_subset = N > maxlines ? sample(1:N, maxlines, replace=false) : 1:N
    p = plot(legend=false, xlabel="Dimension", ylabel="Value")

    for row in eachrow(data)
        plot!(p, 1:M, row, alpha=alpha_val)
    end
    return p
end

# 7. Plot
#pre_synapses = [pre for pre in pre_synapses if length(pre)!=0]
#post_synapses = [post for post in post_synapses if length(post)!=0]


# Filter out empty lists
pre_synapses = [pre for pre in pre_synapses if length(pre) != 0]
post_synapses = [post for post in post_synapses if length(post) != 0]
#sample_size = min(length(pre_synapses), length(post_synapses))  # or define a fixed number
#sample_size = min(length(pre_synapses), length(post_synapses))  # or define a fixed number


#sample_size = length(pre_synapses)
# Subsample from both pre and post synapses
#pre_synapses_sample = rand(pre_synapses, sample_size)
#post_synapses_sample = rand(post_synapses, sample_size)

function dothis(pre_synapses,post_synapses)
    including_smaller_sets_pre = []
    including_smaller_sets_post = []

    @showprogress for (i,j) in zip(pre_synapses,post_synapses)

        push!(including_smaller_sets_pre,rand(i, Int(trunc(length(i)/10))))
        push!(including_smaller_sets_post,rand(j, Int(trunc(length(j)/10))))
    end    
    p = Plots.plot()
    cnt=0
    @showprogress for (ilist,jlist) in zip(including_smaller_sets_pre,including_smaller_sets_post)
        for (i,j) in zip(ilist,jlist)
            Plots.plot!(p,cnt,[i,j],legend=false)
            
            #gui(Plots.plot!(p,cnt,[i,j],legend=false))
        end
        cnt+=2
    end
    return p
end
p = dothis(pre_synapses,post_synapses)
savefig("update2.png")
#=
cnt=0

for (ilist,jlist) in zip(pre_synapses,post_synapses)
    for (i,j) in zip(ilist,jlist)
        Plots.plot!(p,cnt,[i,j],legend=false)
        savefig("update.png")
        #gui(Plots.plot!(p,cnt,[i,j],legend=false))
    end
    cnt+=2
end
+#

data0 = Matrix([pre_synapses[1] post_synapses[1]])
data1 = Matrix([pre_synapses[2] post_synapses[2]])

#p = parallelcoordsB(data1)
p = parallelcoords_ragged([pre_synapses[1], post_synapses[1]])

savefig("blah.png")
    # Number of rows you want to keep
    #k = 5_000  # for example

    # Pick k unique row indices from 1:N
    #rows_subset = sample(1:N, k, replace=false)

    # Create the smaller matrix
    #data_subset = data[rows_subset, :]
    #=
    function parallelcoords(data::AbstractMatrix; colnames=String[])
        # Number of observations (N) and number of dimensions (M)
        N, M = size(data)
        
        # Normalize each column to [0, 1] (common approach for parallel coords)
        #   This helps compare columns even if they have different scales
        data_min = mapslices(minimum, data; dims=1)
        data_max = mapslices(maximum, data; dims=1)
        #data_norm = (data .- data_min) ./ (data_max .- data_min)
        
        # Prepare the plot
        p = Plots.plot(legend = false, xlabel="Dimension", ylabel="Normalized Value")
        
        # If column names are provided, use them on the x-axis
        #if length(colnames) == M
        #   xticks_vals = 1:M
        #  xticks_labels = colnames
        #   p.xticks = (xticks_vals, xticks_labels)
        #end
        
        # Add a line for each observation
        for i in 1:N
            plot!(p, 1:M, data[i, :], alpha=0.3)
        end
        
        return p
    end
    =#


    #data = Matrix([pre_synapses[1] post_synapses[1]])

    # Mock data
    #data = data_subset#rand(1000, 6)




# Example usage:
#p = parallelcoords(data_subset; colnames=["A", "B"])
#display(p)
#=





using Plots, Statistics, Loess
v1 = data[1,:][1]  # e.g., [1, 43, 93, 136, 184, ...]
v2 = data[2,:][1]  # e.g., [1, 48, 99, 152, 214, ...]

# Assuming v1 and v2 are your data vectors (already defined)

# Define the number of segments
num_segments = 5
segment_length = div(length(v1), num_segments)  # segment length
means_v1 = Float64[]
means_v2 = Float64[]
x_positions = Int[]

for i in 1:num_segments
    start_index = (i - 1) * segment_length + 1
    end_index = i * segment_length > length(v1) ? length(v1) : i * segment_length
    
    push!(means_v1, mean(v1[start_index:end_index]))
    push!(means_v2, mean(v2[start_index:end_index]))
    push!(x_positions, i - 1)  # Default x position based on segments
end

# Create plot with segmented means
p = plot(x_positions, means_v1, seriestype=:line, color=:blue, label="Segmented Vector 1", linewidth=2)
plot!(x_positions, means_v2, seriestype=:line, color=:red, label="Segmented Vector 2", linewidth=2)

 p = scatter(
           fill(0, length(v1)),  # x-values are all zero
           v1,                   # y-values come from the first vector
           label = "Vector 1",
           markersize = 0.25,
           color = :blue
       )

       # Add the second vector at x=1
 scatter!(
           fill(1, length(v2)),  # x-values are all one
           v2,
           label = "Vector 2",
           markersize = 0.25,
           color = :red
       )

       # Now draw lines connecting each pair (parallel coordinates lines)
 for i in 1:length(v1)
           plot!([0, 1], [v1[i], v2[i]],
                 color = :black,
                 label = false,
                 alpha = 0.3,
                 markersize = 0.2  # slightly transparent lines
           )
       end
gui(p)
=#

#gui(p)
#Plot{Plots.GRBackend() n=5}

#julia> display(Plots.plot(p))
#Plot{Plots.GRBackend() n=5}

#julia> for i in 1:size(data, 1)
#    display(Plots.plot!(p, 1:3, data[i, :], label="", color=:black, marker=:circle))
#end
#n = 5
#k = 20

#data = [randn(k) .* (rand() + 1) * 10 for _ in 1:n]

TotalNNeurons = sum([v.N for v in values(neurons)])

duration = 15000ms
#SNN.monitor([model.pop...], [:fire])
#SNN.monitor([model.pop...], [:v], sr=200Hz)
#SNN.sim!(model=model; duration = duration, pbar = true, dt = 0.125)
#display(SNN.raster(model.pop, [10s, 15s]))
#display(SNN.raster(model.pop, [14.5s, 15s]))

#Trange = 5s:10:15s
#frE, interval, names_pop = SNN.firing_rate(model.pop, interval = Trange)
#display(plot(interval, mean.(frE), label=hcat(names_pop...), xlabel="Time [ms]", ylabel="Firing rate [Hz]", legend=:topleft))
##

#display(vecplot(model.pop.E23, :v, neurons =1, r=0s:15s,label="soma"))
#layer_names, conn_probs, conn_j = potjans_conn(4000)
#pj = heatmap(conn_j, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:bluesreds,  title="Synaptic weights", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500), clims=(-maximum(abs.(conn_j)), maximum(abs.(conn_j))))
#using DataFrames
#variables = DataFrames.names(data)

# Define source and target nodes
# Create the plot
#plot = PlotlyJS.plot(sankey)
#display(plot)

#using Plots
#=
# Assuming 'data' is already defined and holds your data
n = size(data, 1)

# Create x positions for the axis
x_positions = range(1, stop=size(data, 2), length=size(data, 2))

# Create an initial plot
p = plot(x_positions, zeros(n), ylabel="Value", xlabel="Variables", title="Your Plot Title")

# Plot lines connecting the points for each row
for i in 1:n
    plot!(p, x_positions, data[i, :], color=:black, linewidth=0.8)
end

# Mark points
for i in 1:n
    scatter!(p, x_positions, data[i, :], markersize=5)
    @show(data[i,:])
end

# Display the plot
gui(p)
=#
#=
let
    s = Scene(camera = campixel!)

    n = 5
    k = 20
    data = data2
    #data = [randn(k) .* (rand() + 1) * 10 for _ in 1:n]


    limits = extrema.(data)

    scaled = [(d .- mi) ./ (ma - mi) for (d, (mi, ma)) in zip(data, limits)]

    width = 600
    height = 400
    offset = 100

    for i in 1:n
        x = (i - 1) / (n - 1) * width
        MakieLayout.LineAxis(s, limits = limits[i],
            spinecolor = :black, labelfont = "Arial",
            ticklabelfont = "Arial", spinevisible = true,
            minorticks = IntervalsBetween(2),
            endpoints = Makie.Point2f0[(offset + x, offset), (offset + x, offset + height)],
            ticklabelalign = (:right, :center), labelvisible = false)
    end

    for i in 1:k
        values = map(1:n, data, limits) do j, d, l
            x = (j - 1) / (n - 1) * width
            Makie.Point2f0(offset + x, (d[i] - l[1]) ./ (l[2] - l[1]) * height + offset)
        end

        lines!(s, values, color = get(Makie.ColorSchemes.inferno, (i - 1) / (k - 1)),
            show_axis = false)
    end

    s
end
=#

#using SpikingNeuralNetworks:MetaGraphs
#my_graph = SNN.graph(conn_j)
##


##
set_theme!(theme_light())
f, ax, p = graphplot(my_graph, 
                    edge_width=[0.1 for i in 1:ne(my_graph)],
                     node_size=[30 for i in 1:nv(my_graph)],
                     arrow_shift=0.90,
                     nlabels=[get_prop(my_graph, v, :name) for v in vertices(my_graph)],
                     nlabels_distance=20,
                     )

    

                # f, ax, p = graphplot(my_graph, n_labels=names, nlabels_fontsize=12,node_size = 30, edge_width = .1, arrow_shift=.90)
# hidedecorations!(ax)
# names = [get_prop(my_graph, v, :name) for v in vertices(my_graph)]
deregister_interaction!(ax, :rectanglezoom)
register_interaction!(ax, :nhover, NodeHoverHighlight(p))
register_interaction!(ax, :ehover, EdgeHoverHighlight(p))
register_interaction!(ax, :ndrag, NodeDrag(p))
register_interaction!(ax, :edrag, EdgeDrag(p))

#pprob=heatmap(conn_probs, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:viridis,  title="Connection probability", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500))
#display(plot(pprob, pj, layout=(1,2), size=(1000,500), margin=Plots.mm))
##
