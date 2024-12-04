##
using MLJ
using LIBSVM
using StatsBase
using Statistics


model_info = (repetition=200, 
            peak_rate=8.0,
            proj_strength=20.0,
            p_post = 0.08
            )
path = datadir("sequence_recognition", "overlap_lexicon")
data = load_data(path, "recall_phase", model_info)
@unpack model, seq, mytime, lexicon = data

##
function get_intervals(model, interval=[-50ms, 100ms])
    N = model.pop.E.N
    offsets = Vector{Vector{Float32}}()
    ys = Vector{Symbol}()
    for word in seq.symbols.words
        for myinterval in sign_intervals(word, seq)
            offset = myinterval[end] .+ interval
            push!(offsets, offset)
            push!(ys, word)
        end
    end
    return offsets, ys, N
end

function train(Xs, ys)
    Xs = Xs .+ 1e-1
    Xtrain = Xs[:, 1:2:end]
    Xtest  = Xs[:, 2:2:end]
    ytrain = string.(ys[1:2:end])
    ytest  = string.(ys[2:2:end])
    ZScore = fit(StatsBase.ZScoreTransform,Xtrain, dims=2)
    Xtrain = StatsBase.transform(ZScore, Xtrain)
    Xtest = StatsBase.transform(ZScore, Xtest)

    classifier = svmtrain(Xtrain, ytrain)
    # Test model on the other half of the data.
    ŷ, decision_values = svmpredict(classifier, Xtest);
    @printf "Accuracy: %.2f%%\n" mean(ŷ .== ytest) * 100
end

offsets, ys, N = get_intervals(model)

FR = zeros(N, length(ys))
Threads.@threads for i in eachindex(offsets)
    offset = offsets[i]
    FR[:,i] = length.(spiketimes(model.pop.E, interval = offset))
end

M = zeros(N, length(ys))
membrane, r_v = SNN.interpolated_record(model.pop.E, :v_s)
Threads.@threads for i in eachindex(offsets)
    offset = offsets[i]
    offset[end] > r_v[end] && continue
    range = offset[1]:1ms:offset[2]
    M[:,i] = mean(membrane[:, range], dims=2)[:,1]
end

train(M, ys)
train(Xs, ys)


average_activity = zeros(length(seq.symbols.words), N)
for i in 1:length(seq.symbols.words)
    word = seq.symbols.words[i]
    ids =  findall(==(word), ys)
    average_activity[i,:] = mean(Xs[:,ids], dims=2)[:,1]
end

heatmap(cor(average_activity'), ticks=(1:length(seq.symbols.words), seq.symbols.words), c=:amp, clims=(0.6,1))