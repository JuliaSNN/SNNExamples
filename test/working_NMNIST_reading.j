using Plots
using JLD2 
@load "labeled_packets.jld2" population_code time_and_offset

gui(Plots.scatter(time_and_offset,population_code))


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

