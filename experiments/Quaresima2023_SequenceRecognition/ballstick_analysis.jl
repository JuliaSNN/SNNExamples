using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using LaTeXStrings
using YAML
using MLJ
##
model_info = (repetition=50, 
            peak_rate=8.0,
            proj_strength=20.0,
            p_post = 0.05,
            UUID = "nNBR"
            )

root = YAML.load_file("conf.yml")["paths"]["zeus"]
path = joinpath(root, "sequence_recognition", "overlap")
# data = load_data(path, "Tripod-associative", model_info)
path= "/Users/cocconat/Documents/Research/projects/network_models/data/sequence_recognition/overlap/associative-UUID=a25b-p_post=0.05-peak_rate=8.0-proj_strength=20.0-repetition=40.data.jld2"
data = load_data(model_path)
@unpack model, seq, mytime, lexicon = data

offsets, ys = all_intervals(:words, seq, interval=[0ms, 100ms])
@unpack N = model.pop.E
##

## Compute the confusion matrix of the most active population in the interval [0ms, 100ms]
_confusion_matrix = score_activity(model, seq, [0ms, 100ms], targets=[:d1, :d2])
heatmap(_confusion_matrix, c=:amp, clims=(0,1), xlabel="True", ylabel="Predicted", ticks=(1:length(seq.symbols.words), seq.symbols.words), size=(500,500), xrotation = 45)

## Compute the spike count features and the membrane potential features and use them to train a SVM classifier
S = spikecount_features(model.pop.E, offsets)
M = sym_features(:v_s, model.pop.E, offsets)
(SVCtrain(S, ys; seed=123, p=0.1) - 0.1) / 0.9

## Plot the activity of the network, and the word and phoneme raster plots
T = get_time(mytime)
Trange = T-0.5s:1ms:T-100ms
names, pops = filter_items(model.stim) |> subpopulations
pr1 = SNN.raster(model.pop.E, Trange, populations=pops, names=names)
pr2 = SNN.raster(model.pop, Trange)
plot(pr1, pr2, layout = (2, 1), size = (800, 800), margin = 5Plots.mm)
plot_activity(model, Trange)

## Raster plot the activity of the neurons in certain population
pop = model.stim.LOP_d.neurons
vecplot(model.pop.E, :v_d, r=Trange, neurons=pop, pop_average=true, ribbon=false)
raster(spiketimes(model.pop.E, interval=Trange)[pop], Trange)

## Plot the firing rate of a single population
neurons = model.stim.POLL_d.neurons
fr, interval = firing_rate(spiketimes(model.pop.E)[neurons], τ =10ms)
plot(interval, mean(fr, dims=1)[1,:], xlabel="Time (ms)", ylabel="Firing rate (Hz)", title="Firing rate of :L_d")
plot!(xlims=(1000,2500))


## Plot the average activity of the network

plot_average_word_activity(:fire, :POLLEN, model, seq, zscore=false, target=:d)
plot_average_word_activity(:v_s, :POLLEN, model, seq, zscore=false, target=:d)
plot_average_word_activity(:fire, :POLLEN, model, seq, zscore=false, target=:d)

## Plot the average activity of the network for all words
using ThreadTools
plots_fire = tmap(seq.symbols.words) do word
    plot_average_word_activity(:fire, word, model, seq, before=100ms, target = :d, )
end
plots_mem = tmap(seq.symbols.words) do word
    plot_average_word_activity(:v_s, word, model, seq, before=100ms, target = :d, )
end
plot(plots_fire..., layout=(5,2), size=(800,1300), margin=5Plots.mm, legend=false)
plot(plots_mem..., layout=(5,2), size=(800,1300), margin=5Plots.mm, legend=false)

##  Plot the average synaptic weight between populations§
names, pops = filter_items(model.stim) |> subpopulations
connections = zeros(length(pops), length(pops))
w_indices = Int64[]
for pre in eachindex(pops)
    for post in eachindex(pops)
        pre_pop = pops[pre]
        post_pop = pops[post]
        # update_weight!(pre_pop, post_pop, model.syn.E_to_E)
        connections[post, pre] = 0.5*((average_weight(pre_pop, post_pop, model.syn.E_to_E1) - mean(model.syn.E_to_E1.W))./mean(model.syn.E_to_E1.W) +(average_weight(pre_pop, post_pop, model.syn.E_to_E2) - mean(model.syn.E_to_E2.W))./mean(model.syn.E_to_E2.W))*100
        append!(w_indices , weights_indices(pre_pop, post_pop, model.syn.E_to_E1))
    end
end
plasticity!(model.syn.norm1, model.syn.norm1.param)
connections 
heatmap(ticks=(1:length(names),names), connections, c=:bluesreds, xlabel="Pre-synaptic population", ylabel="Post-synaptic population", title="Average synaptic weight "*L"(w_0)", clims=(-maximum(connections),maximum(connections)), size=(500,400), xrotation=45)

vline!([10.5], c=:black, ls=:dash, label="")
hline!([10.5], c=:black, ls=:dash, label="")


## Plot the synaptic weights from L and P to POLL
@unpack stim = model
histogram(matrix(model.syn.E_to_E)[stim.POLL_d.neurons, model.stim.L_d.neurons][:], alpha=0.5, bins=1:0.5:45, label="L->POLL")
histogram!(matrix(model.syn.E_to_E)[stim.POLL_d.neurons, model.stim.P_d.neurons][:], alpha=0.5, bins=1:0.5:45, label="P->POLL")
plot!(title = "Synaptic weights from L and P to POLL")



## Plot the learning curve of the network
names, pops = filter_populations(model.stim) |> subpopulations
indices_words = Int64[]
indices_phonemes = Int64[]
indices_phtowords = Int64[]
indices_wordtoph = Int64[]
@unpack phonemes, words = symbolnames(seq)
for pre in eachindex(pops)
    for post in eachindex(pops)
        pre_pop = pops[pre]
        post_pop = pops[post]
        if names[pre] ∈ phonemes && names[post] ∈ words
            !(occursin(names[pre], names[post])) && continue
            append!(indices_phtowords , weights_indices(pre_pop, post_pop, model.syn.E_to_E))
        elseif names[pre] ∈ words && names[post] ∈ phonemes
            !(occursin(names[post], names[pre])) && continue
            append!(indices_wordtoph , weights_indices(pre_pop, post_pop, model.syn.E_to_E))
        elseif names[pre] ∈ phonemes && names[post] ∈ phonemes
            !(names[post] ==  names[pre]) && continue
            append!(indices_phonemes , weights_indices(pre_pop, post_pop, model.syn.E_to_E))
        elseif names[pre] ∈ words && names[post] ∈ words
            !(names[post] ==  names[pre]) && continue
            append!(indices_words , weights_indices(pre_pop, post_pop, model.syn.E_to_E))
        else 
            throw("No connection found")
        end
    end
end


##
W, r_v = SpikingNeuralNetworks.interpolated_record(model.syn.E_to_E, :W)
r = range(r_v[1], r_v[end], length = 1000)
control_indices = setdiff(Set(axes(W, 1)), Set(vcat(indices_phonemes, indices_words, indices_phtowords, indices_wordtoph))) |> collect
ave_weights = [mean(W[indices, r], dims=1)[1,:] for indices in [indices_phonemes, indices_words, indices_phtowords, indices_wordtoph, control_indices]]
std_weights = [ std(W[indices, r], dims=1)[1,:] for indices in [indices_phonemes, indices_words, indices_phtowords, indices_wordtoph, control_indices]]


labels = ["Phonemes to Phonemes" "Words to Words" "Phonemes to Words" "Words to Phonemes"]
plot(
    plot(mean(W, dims=1)[1,:], label="Average weight", xlabel="Time (ms)", ylabel="Weight"),
    plot(r,ave_weights, ribbon = std_weights'./5, label="", xlabel="Time (ms)", ylabel="Weight", labels= labels),
    layout = (2, 1),
    size = (500, 800),
    margin = 5Plots.mm
)
##

