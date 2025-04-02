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

root = YAML.load_file(projectdir("conf.yml"))["paths"]["local"]
path = joinpath(root, "sequence_recognition", "overlap")

include(projectdir("examples/parameters/dendritic_network.jl"))

exp_config = (      # Sequence parameters
                    init_silence=1s, 
                    repetition=50, 
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
                    STDP = false,
        )

model_info = (repetition=exp_config.repetition, 
            peak_rate=exp_config.peak_rate,
            proj_strength=exp_config.proj_strength,
            p_post = exp_config.p_post,
            UUID = randstring(4)
            )

# Merge network and stimuli in model
model = tripod_network(;exp_config...)
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d1, :v_d2, :v_s], sr=200Hz)
SNN.train!(model=model, duration= 10s, pbar=true, dt=0.125)
model_path = save_model(path=path, name="Tripod_no_inputs", model=model, info=model_info, config=exp_config)
#

vecplot(model.pop.E, [:v_s], interval=9s:10s, neurons=10:20, pop_average=true)
fr, r, labels = firing_rate(model.pop, interval=5s:10s, pop_average=true, τ=10ms)
p1 = plot(r, fr, label=hcat(labels...), xlabel="Time [s]", ylabel="Firing rate [Hz]", title="Firing rate", legend=:topleft)
p2 = raster(model.pop, 5s:10s)
plot(p1, p2, layout=(2,1), size=(800, 600))


