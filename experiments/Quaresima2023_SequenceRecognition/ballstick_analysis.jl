using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using YAML
##
model_info = (repetition=50, 
            peak_rate=8.0,
            proj_strength=20.0,
            p_post = 0.05,
            UUID = "nNBR"
            )

root = YAML.load_file("conf.yml")["paths"]["zeus"]
path = joinpath(root, "sequence_recognition", "overlap")
data = load_data(path, "Tripod-associative", model_info)
data = load_data("data/sequence_recognition/overlap_lexicon/associative_phase-p_post=0.08-peak_rate=8.0-proj_strength=20.0-repetition=200.data.jld2")
@unpack model, seq, mytime, lexicon = data

offsets, ys = all_intervals(:words, seq, interval=[0ms, 100ms])
@unpack N = model.pop.E
##

## Compute the confusion matrix of the most active population in the interval [0ms, 100ms]
confusion_matrix = score_activity(model, seq, [0ms, 100ms])
heatmap(confusion_matrix, c=:amp, clims=(0,1), xlabel="True", ylabel="Predicted", ticks=(1:length(seq.symbols.words), seq.symbols.words), size=(500,500), xrotation = 45)

## Compute the spike count features and the membrane potential features and use them to train a SVM classifier
S = spikecount_features(model.pop.E, offsets)
M = sym_features(:v_s, model.pop.E, offsets)
(SVCtrain(S, ys; seed=123, p=0.1) - 0.1) / 0.9

## Plot the activity of the network, and the word and phoneme raster plots
T = get_time(mytime)
Trange = T-0.5s:1ms:T-100ms
names, pops = filter_populations(model.stim) |> subpopulations
pr1 = SNN.raster(model.pop.E, Trange, populations=pops, names=names)
pr2 = SNN.raster(model.pop, Trange)
plot(pr1, pr2, layout = (2, 1), size = (800, 800), margin = 5Plots.mm)
plot_activity(model, Trange)

## Raster plot the activity of the cells in certain population
pop = model.stim.LOP_d.cells
vecplot(model.pop.E, :v_d, r=Trange, neurons=pop, pop_average=true, ribbon=false)
raster(spiketimes(model.pop.E, interval=Trange)[pop], Trange)

## Plot the firing rate of a single population
cells = model.stim.POLL_d.cells
fr, interval = firing_rate(spiketimes(model.pop.E)[cells], τ =10ms)
plot(interval, mean(fr, dims=1)[1,:], xlabel="Time (ms)", ylabel="Firing rate (Hz)", title="Firing rate of :L_d")
plot!(xlims=(1000,2500))


## Plot the average activity of the network
function plot_average_activity(sym, word, model, seq; before=100ms, after=300ms, zscore=true)
    membrane, r_v = SNN.interpolated_record(model.pop.E, sym)
    myintervals = sign_intervals(word, seq)
    Trange = -before:1ms:diff(myintervals[1])[1]+after
    activity = zeros(length(seq.symbols.words),size(Trange,1))
    for w in eachindex(seq.symbols.words)
        cells = getcells(model.stim, seq.symbols.words[w], :d)
        ave_fr = mean(membrane[cells, :])
        std_fr = std(membrane[cells, :])
        n = 0
        for myinterval in myintervals
            _range = myinterval[1]-before:1ms:myinterval[2]+after
            _range[end] > r_v[end] && continue
            v = mean(membrane[cells, _range], dims=1)[1,:]
            activity[w, :] += zscore ? (v .- ave_fr)./std_fr : v
            n+=1
        end
        activity[w, :] ./= n
    end
    plot(Trange, activity[:,:]', label=hcat(string.(seq.symbols.words)...), xlabel="Time (ms)", ylabel="Membrane potential (mV)", title="")
    vline!([0, diff(myintervals[1])[1]], c=:black, ls=:dash, label="")
    word_id = findfirst(seq.symbols.words .== word)
    plot!(Trange, activity[word_id,:], c=:black, label=string(word), lw=5, )
end

plot_average_activity(:v_s, :POLLEN, model, seq, zscore=false)

## Plot the average activity of the network for all words
using ThreadTools
plots_fire = tmap(seq.symbols.words) do word
    plot_average_activity(:v_s, word, model, seq, before=100ms,)
end
plots_mem = tmap(seq.symbols.words) do word
    plot_average_activity(:v_s, word, model, seq, before=100ms)
end
plot(plots_fire..., layout=(5,2), size=(800,1300), margin=5Plots.mm, legend=false)
plot(plots_mem..., layout=(5,2), size=(800,1300), margin=5Plots.mm, legend=false)

##  Plot the average synaptic weight between populations§
names, pops = filter_populations(model.stim) |> subpopulations
connections = zeros(length(pops), length(pops))
indices = Int64[]
for pre in eachindex(pops)
    for post in eachindex(pops)
        pre_pop = pops[pre]
        post_pop = pops[post]
        # update_weight!(pre_pop, post_pop, model.syn.E_to_E)
        connections[post, pre] = (average_weight(pre_pop, post_pop, model.syn.E_to_E) - mean(model.syn.E_to_E.W))./mean(model.syn.E_to_E.W)
        append!(indices , weights_indices(pre_pop, post_pop, model.syn.E_to_E))
    end
end
plasticity!(model.syn.norm, model.syn.norm.param)
using LaTeXStrings
heatmap(ticks=(1:length(names),names), connections, c=:amp, xlabel="Pre-synaptic population", ylabel="Post-synaptic population", title="Average synaptic weight "*L"(w_0)", size=(500,400), xrotation=45)
vline!([10.5], c=:black, ls=:dash, label="")
hline!([10.5], c=:black, ls=:dash, label="")


## Plot the synaptic weights from L and P to POLL
@unpack stim = model
histogram(matrix(model.syn.E_to_E)[stim.POLL_d.cells, model.stim.L_d.cells][:], alpha=0.5, bins=1:0.5:45, label="L->POLL")
histogram!(matrix(model.syn.E_to_E)[stim.POLL_d.cells, model.stim.P_d.cells][:], alpha=0.5, bins=1:0.5:45, label="P->POLL")
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

