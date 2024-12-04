using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

model_info = (repetition=200, 
            peak_rate=8.0,
            proj_strength=20.0,
            p_post = 0.08
            )
path = datadir("sequence_recognition", "overlap_lexicon")

load_model(path, "recall_phase", model_info)
model_path = datadir("sequence_recognition", "overlap_lexicon", name) |> path -> (mkpath(dirname(path)); path) 

@unpack model, seq, mytime, lexicon = copy_model(model_path)
@unpack model, seq, mytime, lexicon = load_model(model_path)
##
T = get_time(mytime)
Trange = T-1s:1ms:T-100ms
names, pops = filter_populations(model.stim) |> subpopulations
pr1 = SNN.raster(model.pop.E, Trange, populations=pops, names=names)
pr2 = SNN.raster(model.pop, Trange)
plot(pr1, pr2, layout = (2, 1), size = (800, 800))
# plot_activity(model, Trange)
## Target activation with stimuli
# Trange = 98000ms:1ms:100000ms

# control = :POLLEN

##
using LaTeXStrings
word = :GOLDEN
myintervals = sign_intervals(word, seq)
membrane, r_v = SNN.interpolated_record(model.pop.E, :v_s)
membrane, r_v =  firing_rate(model.pop.E, τ=20ms)
Trange = -500ms:1ms:diff(myintervals[1])[1]+500ms
activity = zeros(length(seq.symbols.words),size(Trange,1))
for w in eachindex(seq.symbols.words)
    cells = getcells(model.stim, seq.symbols.words[w], :d)
    n = 0
    for myinterval in myintervals
        _range = myinterval[1]-500ms:1ms:myinterval[2]+500ms
        if _range[end] < r_v[end]
            activity[w, :] += mean(membrane[cells, _range], dims=1)[1,:]
            n+=1
        end
    end
    activity[w, :] ./= n
end
plot(Trange, activity[:,:]', label=hcat(string.(seq.symbols.words)...), xlabel="Time (ms)", ylabel="Membrane potential (mV)", title="")
vline!([0, diff(myintervals[1])[1]], c=:black, ls=:dash, label="")
word_id = findfirst(seq.symbols.words .== word)
plot!(Trange, activity[word_id,:], c=:black, label=string(word), lw=5, )

##

# @unpack stim = model
# pv = plot()
# cells = getcells(stim, word, :d)
# control_cells = getcells(stim, control, :d)

# SNN.vecplot!(pv, model.pop.E, :v_d, r = Trange, neurons = control_cells, dt = 0.125, pop_average = true, c = :grey, lw=2, ls=:dot)
# SNN.vecplot!(pv, model.pop.E, :v_d, r = Trange, neurons = cells, dt = 0.125, pop_average = true, lw=3)
# vline!(myintervals./1000, c = :black, ls=:dash)
# plot!(title = "Depolarization of :AB vs :BA with stimuli AB")
# plot!(xlabel = "Time (s)", ylabel = "Membrane potential (mV)")
# plot!(margin = 5Plots.mm, ylims=:auto)
# # plot(pv, pr, layout = (2, 1), size = (800, 800))
# ##

# ##
# %%
@unpack stim = model
histogram(matrix(model.syn.E_to_E)[stim.POLL_d.cells, model.stim.L_d.cells][:], alpha=0.5, bins=1:0.5:45, label="L->POLL")
histogram!(matrix(model.syn.E_to_E)[stim.POLL_d.cells, model.stim.P_d.cells][:], alpha=0.5, bins=1:0.5:45, label="P->POLL")
plot!(title = "Synaptic weights from L and P to POLL")
# average_weight(stim.A_d.cells, model.stim.AB_d.cells, model.syn.E_to_E)
# average_weight(stim.AB_d.cells, model.stim.AB_d.cells, model.syn.E_to_E)

##
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


## 
## Recall phase
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

