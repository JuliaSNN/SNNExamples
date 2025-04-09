using DrWatson
using Plots
using SpikingNeuralNetworks
using Distributions
SNN.@load_units
using YAML
##
SETTINGS = YAML.load_file(projectdir("conf.yml"))
begin
    root = SETTINGS["paths"]["local"]
    include("parameters.jl")
    include("model.jl")
    plots_path = plotsdir("LKD2014") |> mkpath
    data_path = datadir("LKD2014") |> mkpath  
end
config = let
    network = :tripod
    seed = 123
    rate_factor= 300
    input_neurons = 1
    sounds = "only_tones" # "chirps_tones", "only_tones", "all_sounds"
    input_config = (rate_factor=rate_factor, 
                    presentations=150,
                    sounds_train=50,
                    input_neurons = input_neurons,
                    init_silence=30s,
                    between_silence=0.5s,
                    sound_offset=250ms,
                    network=network,
                    sounds = sounds,
                    store_model = false,
                    sounds_path = datadir("zeus", "ExpData") |> mkpath,
                    data_path = data_path,
                    trials_path = joinpath(data_path, "trials") |> mkpath,
                    seed = seed) 

    network_config = (;LKD_network...,
                        silent = true,
                        input_STDP = false,
                        recurrent_STDP = false)

    info = (;rate_factor, input_neurons, presentations = input_config.presentations, network, seed, input_STDP = network_config.input_STDP, recurrent_STDP=network_config.recurrent_STDP, sounds)
    config = (network=network_config, inputs=input_config, info=info)
end

#
@unpack network, seed, sounds, rate_factor, input_neurons = config.inputs
@unpack data_path, sounds_path, store_model = config.inputs
n_neurons = 1000
model = soma_network(config.network)
#
train!(;model, duration = 10s, pbar=true)

raster(model.pop, 5s:10s)
fr, r, labels = firing_rate(model.pop, interval=5s:10s, pop_average=true, τ=10ms)
plot(r, fr)