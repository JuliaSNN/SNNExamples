using DrWatson
quickactivate("../../")
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Distributions
using Random
using Statistics
using YAML
##

root = YAML.load_file(projectdir("conf.yml"))["paths"]["zeus"]
path = joinpath(root, "sequence_recognition", "overlap")

include(projectdir("examples/parameters/dendritic_network.jl"))
lexicon = let
    dictionary = getdictionary(["POLLEN", "GOLD", "GOLDEN", "DOLL", "LOP", "GOD", "LOG", "POLL", "GOAL", "DOG"])
    duration = getduration(dictionary, 50ms)
    config_lexicon = (ph_duration=duration, dictionary=dictionary)
    lexicon = generate_lexicon(config_lexicon)
end

exp_config = (      # Sequence parameters
                    init_silence=1s, 
                    repetition=100, 
                    silent_intervals=1, 
                    peak_rate=8kHz, 
                    start_rate=8kHz, 
                    decay_rate=10ms,
                    proj_strength=20pA,
                    p_post = 0.05f0,
                    targets= [:d1, :d2],
                    words=true,
                    # Network parameters
                    NE = 1200,
                    name =  "bursty_dendritic_network",
                    params = bursty_dendritic_network,
                    STDP = true,
        )

model_info = (repetition=exp_config.repetition, 
            peak_rate=exp_config.peak_rate,
            proj_strength=exp_config.proj_strength,
            p_post = exp_config.p_post,
            UUI = randstring(4)
            )

## Merge network and stimuli in model
network = tripod_network(;exp_config...)
model = network
stim, seq = SNNUtils.step_input_sequence(network = network, lexicon = lexicon; exp_config..., )
model = merge_models(network, stim)
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d1, :v_d2, :v_s], sr=200Hz)
SNN.monitor([model.syn...], [ :W], sr=1Hz)
SNN.train!(model=model, duration= 10s, pbar=true, dt=0.125)
model_path = save_model(path=path, name="Tripod-associative", model=model, info=model_info, lexicon=lexicon, config=exp_config)
model_path = save_model(path=path, name="Tripod-associative", model=model, info=model_info, lexicon=lexicon, config=exp_config, seq=seq)
##
model_path = get_path(;path, name="Tripod-associative", info=model_info)
@unpack model, seq, mytime, lexicon, config = load_model(model_path)
recall_config = (;config..., STDP=false, words=false,)
seq = randomize_sequence!(;seq=seq, model=model, recall_config...)
model.syn.E_to_E.param.active[1] = recall_config.STDP
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
duration = sequence_end(seq)
mytime = SNN.Time()
SNN.train!(model=model, duration= duration, pbar=true, dt=0.125, time=mytime)

data = (@strdict seq mytime lexicon recall_config) |> dict2ntuple
path = datadir("sequence_recognition", "overlap_lexicon")
save_model(path=path, name="Tripod-recall", model =model, info=recall_config; data...)
filesize(model_path) |> Base.format_bytes
basename(model_path)


##

raster(model.pop, 1s:1ms:10s, every=1, size=(1200, 1000))