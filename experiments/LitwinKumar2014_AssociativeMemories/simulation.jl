# Litwin-Kumar & Doiron (2014) network (see model.jl and parameters.jl) run for 3 s (the original ran 10 s; the
# synaptic normalisation makes SNNModels 2.1.0 about 100 times slower than the rest of the network, see STATUS.md)
# without sound input: raster and population rates of the last 2 s. The configuration of the auditory protocol (sounds, presentations,
# silences) that the original script defined is kept in `config` but not used here; the sound files it refers to
# (data/zeus/ExpData) are not part of this repository.
# Runtime: about 3 min.
using DrWatson
using SpikingNeuralNetworks
using Distributions
SNN.@load_units
using CairoMakie
##
begin
    include("parameters.jl")
    include("model.jl")
    data_path = datadir("LKD2014")
end
config = let
    network = :tripod
    seed = 123
    rate_factor = 300
    input_neurons = 1
    sounds = "only_tones" # "chirps_tones", "only_tones", "all_sounds"
    input_config = (rate_factor = rate_factor,
                    presentations = 150,
                    sounds_train = 50,
                    input_neurons = input_neurons,
                    init_silence = 30s,
                    between_silence = 0.5s,
                    sound_offset = 250ms,
                    network = network,
                    sounds = sounds,
                    store_model = false,
                    sounds_path = datadir("zeus", "ExpData"),
                    data_path = data_path,
                    trials_path = joinpath(data_path, "trials"),
                    seed = seed)

    network_config = (; LKD_network...,
                        silent = true,
                        input_STDP = false,
                        recurrent_STDP = false)

    info = (; rate_factor, input_neurons, presentations = input_config.presentations, network, seed,
            input_STDP = network_config.input_STDP, recurrent_STDP = network_config.recurrent_STDP, sounds)
    (network = network_config, inputs = input_config, info = info)
end

model = soma_network(config.network)
SNN.train!(; model, duration = 3s)

SNN.raster(model.pop, 1s:3s)
fr, r, labels = SNN.firing_rate(model.pop, interval = 1s:10ms:3s, pop_average = true, τ = 10ms)
fig = Figure(size = (600, 350))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Firing rate (Hz)")
for (f, l) in zip(fr, labels)
    lines!(ax, r ./ 1000, vec(f), label = string(l))
end
axislegend(ax)
fig
