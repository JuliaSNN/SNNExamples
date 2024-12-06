using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Distributions
using Random
using Statistics
using Dates
using YAML
##

## Set the paths and get the experiments
root = YAML.load_file(projectdir("conf.yml"))["paths"]["zeus"]
path = joinpath(root, "sequence_recognition", "deviant")
path_generators = "/Users/cocconat/Documents/Research/projects/bio-seqlearn/data/models/tripod/task_generators/deviant"
path_tasks = "/Users/cocconat/Documents/Research/projects/bio-seqlearn/data/models/tripod/tasks/deviant"
experiments = import_bioseq_tasks(path_generators, path_tasks)
## Prepare simulation
# Get the network model from the parameters
network = ballstick_network(;exp_config...)
# Set input parameters
exp_config = (      # Sequence parameters
                    init_silence=0s, 
                    silent_intervals=1, 
                    # Step input parameters
                    peak_rate=8kHz, 
                    start_rate=8kHz, 
                    decay_rate=10ms,
                    proj_strength=20pA,
                    p_post = 0.1f0,
                    targets= [:d],
                    words=true,
                    # Network parameters
                    NE = 1200,
                    name =  "bursty_dendritic_network",
                    params = bursty_dendritic_network,
                    STDP = true,
        )

# Choose one experiment and generate stimuli
exp = experiments[1]
sequence = seq_bioseq(experiment=exp, stage="train")
stim, seq  = step_input_sequence(;seq=sequence, network=network,
                                                experiment=exp, 
                                                exp_config...)

# Merge network and stimuli in model
model = merge_models(network, stim)

## Simulation and training
# Initialize the network with 1 second of silence
train!(model=model, duration= 1s, pbar=true, dt=0.125)

## Simulate and record network activity
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
SNN.monitor(model.syn.E_to_E, [ :W], sr=50Hz)
SNN.train!(model=model, duration= 10s, pbar=true, dt=0.125)
save_model(path=path, name="associative", model=model, info=exp_config, lexicon=lexicon, config=exp_config, mytime=mytime, seq=seq)

## Load the trained model and simulate the recall phase
@unpack model, seq, mytime, lexicon, config = load_model(path, "associative", exp_config)
recall_config = (;config..., STDP=false, words=false,)

sequence = seq_bioseq(experiment=exp, stage="test")
model.syn.E_to_E.param.active[1] = recall_config.STDP
update_sequence!(;seq=sequence, model=model, recall_config...)
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
SNN.train!(model=model, duration= 30s, pbar=true, dt=0.125)
save_model(path=path, name="recall", model=model, info=exp_config, lexicon=lexicon, config=exp_config, mytime=mytime, seq=seq)

## Make raster plots
@unpack model, seq, mytime, lexicon, config = load_data(path, "associative", exp_config)
Trange= 0s:1ms:2s
names, pops = filter_populations(model.stim) |> subpopulations
pr1 = SNN.raster(model.pop.E, Trange, populations=pops, names=names, title="Associative phase")

@unpack model, seq, mytime, lexicon, config = load_data(path, "recall", exp_config)
Trange= 0s:1ms:2s
names, pops = filter_populations(model.stim) |> subpopulations
pr2 = SNN.raster(model.pop.E, Trange, populations=pops, names=names, title="Recall phase")

plot(pr1, pr2, layout = (2, 1), size = (800, 800), margin = 5Plots.mm)