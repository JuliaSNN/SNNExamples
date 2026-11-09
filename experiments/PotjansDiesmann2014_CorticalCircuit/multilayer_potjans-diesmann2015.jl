# Cortical microcircuit of Potjans & Diesmann (2014, Cereb. Cortex 24:785): 8 populations (E and I in layers 2/3,
# 4, 5 and 6, 77,169 neurons at full scale) with the layer-specific connection probabilities of the paper and
# neuron parameters of Litwin-Kumar & Doiron (2014) (AdEx excitatory, IF inhibitory, single-exponential
# synapses). Run at 1 % scale for 15 s with Poisson drive onto the excitatory populations; plots the raster,
# the population rates, and the matrices of connection probability and synaptic weight.
# The neuron parameter set `LKD2014SingleExp` of SNNUtils < 1 no longer exists; it is rebuilt below from the
# Litwin-Kumar & Doiron values (see LitwinKumar2014_AssociativeMemories/parameters.jl); unverified against the original set.
# Runtime: about 60 s.
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
using Statistics, Random
using CairoMakie

# Neuron parameters (Litwin-Kumar & Doiron 2014) with single-exponential synapses
LKD2014SingleExp = (
    AdEx = SNN.AdExParameter(El = -70mV, Vt = -52.0mV, τm = 300pF / 15.0nS, R = 1 / 15.0nS, Vr = -60.0mV),
    PV = SNN.IFParameter(El = -62.0mV, Vr = -57.47mV, Vt = -52.0mV, τm = 20ms, R = 1 / 15.0nS),
    spike = SNN.PostSpike(τabs = 1ms, At = 10mV),
    synapse = SNN.SingleExpSynapse(τe = 6ms, τi = 2ms, E_i = -75mV, E_e = 0mV),
)

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
            neurons[k] = SNN.AdEx(N = Int(v), param = LKD2014SingleExp.AdEx, spike = LKD2014SingleExp.spike,
                                  synapse = LKD2014SingleExp.synapse, name = string(k))
        else
            neurons[k] = SNN.IF(N = Int(v), param = LKD2014SingleExp.PV, synapse = LKD2014SingleExp.synapse, name = string(k))
        end
    end
    return neurons
end

"""
Define Potjans parameters for neuron populations and connection probabilities
"""
function potjans_conn(Ne)

    function j_from_name(pre, post, g)
        if occursin("E", String(pre)) && occursin("E", String(post))
            return g.jee
        elseif occursin("I", String(pre)) && occursin("E", String(post))
            return g.jei
        elseif occursin("E", String(pre)) && occursin("I", String(post))
            return g.jie
        elseif occursin("I", String(pre)) && occursin("I", String(post))
            return g.jii
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

    # !Assign dimensions to these parameters, and some reference
    pree = 0.1
    g = 1.0
    tau_meme = 10.0   # (ms)
    # Synaptic strengths for each connection type
    K = round(Int, Ne * pree)
    sqrtK = sqrt(K)

    # !Same, the convention is j_post_pre
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jee = 0.15 * je
    jie = je
    jei = -0.75 * ji
    jii = -ji
    g_strengths = (; jee, jie, jei, jii)
    
    conn_j = zeros(Float32, size(conn_probs))
    for pre in eachindex(layer_names)
        for post in eachindex(layer_names)
            conn_j[post, pre ] = j_from_name(layer_names[pre], layer_names[post], g_strengths)
        end
    end

    return layer_names, conn_probs, conn_j
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
    @show exc_pop, Ne, Ni
    layer_names, conn_probs, conn_j = potjans_conn(Ne)

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
            s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; conn = (μ = μ, p = p, σ = 0), name = string(pre, "_", post))
            connections[Symbol(string(pre,"_", post))] = s
        end
    end

    ## Create the Poisson stimulus for each population
    stimuli = Dict()
    for pop in exc_pop
        νe = 2.5kHz
        post = neurons[pop]
        s = SNN.PoissonStimulus(post, :ge; param = SNN.PoissonFixed(rate = νe, μ = 1.0f0), neurons = :ALL, name = "PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    return SNN.compose(NamedTuple(neurons), NamedTuple(connections), NamedTuple(stimuli); silent = true)
end


model = potjans_layer(0.01)
duration = 15000ms
SNN.monitor!([model.pop...], [:fire])
SNN.monitor!(model.pop.E23, [:v], sr = 200Hz)
SNN.sim!(model = model; duration = duration, dt = 0.125f0)
SNN.raster(model.pop, [10s, 15s])

frE, interval, names_pop = SNN.firing_rate(model.pop, interval = 5s:10ms:15s, pop_average = true)
fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Firing rate (Hz)")
for (f, l) in zip(frE, names_pop)
    lines!(ax, interval ./ 1000, vec(f), label = string(l))
end
axislegend(ax, position = :lt, nbanks = 2)
fig

##
layer_names, conn_probs, conn_j = potjans_conn(4000)
fig = Figure(size = (1000, 500))
ticks = (1:8, string.(layer_names))
ax1 = Axis(fig[1, 1], xticks = ticks, yticks = ticks, title = "Connection probability", xlabel = "Presynaptic", ylabel = "Postsynaptic", aspect = 1)
hm1 = heatmap!(ax1, conn_probs', colormap = :viridis)
Colorbar(fig[1, 2], hm1)
ax2 = Axis(fig[1, 3], xticks = ticks, yticks = ticks, title = "Synaptic weights", xlabel = "Presynaptic", ylabel = "Postsynaptic", aspect = 1)
m = maximum(abs.(conn_j))
hm2 = heatmap!(ax2, conn_j', colormap = :balance, colorrange = (-m, m))
Colorbar(fig[1, 4], hm2)
fig
