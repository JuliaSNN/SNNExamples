##


model_info = (repetition=200, 
            peak_rate=8.0,
            proj_strength=20.0,
            p_post = 0.08
            )
path = datadir("sequence_recognition", "overlap_lexicon")
data = load_data(path, "recall_phase", model_info)
@unpack model, seq, mytime, lexicon = data

offsets, ys = all_intervals(:words, seq, interval=[-50ms, 100ms])
@unpack N = model.pop.E



M = zeros(N, length(ys))
Threads.@threads for i in eachindex(offsets)
    offset = offsets[i]
    range = offset[1]:1ms:offset[2]
    M[:,i] = mean(membrane[:, range], dims=2)[:,1]
end

M

##
μ = mean(scores)
xs = exp.(-4:0.2:-1.5)
scores = map(xs) do x
    train(M, ys; name= "Membrane potential", seed=123, p=x)
end
k = (scores .- μ) ./ (1-μ)

plot(xs, k, xlabel="Train/Test size", ylabel="Accuracy", legend=false)

##

[train(M, ys; name= "Membrane potential", seed=x) for x in 1:20]


predict(mach, Xnew)train(Xs, ys, "Firing rate")


average_activity = zeros(length(seq.symbols.words), N)
for i in 1:length(seq.symbols.words)
    word = seq.symbols.words[i]
    ids =  findall(==(word), ys)
    average_activity[i,:] = mean(Xs[:,ids], dims=2)[:,1]
end

heatmap(cor(average_activity'), ticks=(1:length(seq.symbols.words), seq.symbols.words), c=:amp, clims=(0.6,1))