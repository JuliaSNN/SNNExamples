using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions
using Dates
##

include(projectdir("examples/parameters/dendritic_network.jl"))
function get_network(;params, name, NE, STDP, kwargs...)
    @unpack exc, pv, sst, plasticity, connectivity=params
    @unpack  noise_params, inh_ratio = params
    exc = quaresima2022
    # pv = duarte2019.PV
    # sst = duarte2019.SST
    # @unpack connectivity, plasticity = quaresima2023
    @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    # Number of neurons in the network
    NI1 = round(Int,NE * inh_ratio.ni1)
    NI2 = round(Int,NE * inh_ratio.ni2)
    # Import models parameters
    # Define interneurons I1 and I2
    # @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    E = SNN.BallAndStickHet(; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="ExcBallStick")
    I1 = SNN.IF(; N = NI1, param = pv, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = sst, name="I2_sst")
    # Define synaptic interactions between neurons and interneurons
    E_to_E = SNN.SpikingSynapse(E, E, :he, :d ; connectivity.EdE..., param= plasticity.vstdp)
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.EIf...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...)
    I2_to_I2 = SNN.SpikingSynapse(I2, I2, :gi; connectivity.IsIs...)
    I2_to_E = SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.EdIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)
    # Define normalization
    norm = SNN.SynapseNormalization(NE, [E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    # background noise
    @unpack exc_soma, exc_dend, inh1, inh2= noise_params
    stimuli = Dict(
        :s   => SNN.PoissonStimulus(E,  :he_s; exc_soma... ),
        :d   => SNN.PoissonStimulus(E,  :he_d; exc_dend... ),
        :i1  => SNN.PoissonStimulus(I1, :ge;   inh1...  ),
        :i2  => SNN.PoissonStimulus(I2, :ge;   inh2...  )
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    # Return the network as a model
    if !STDP
        syn.E_to_E.param.active[1] = false
    end
    merge_models(pop, syn, noise=stimuli, name=name)
end


## Define the network, stimuli and lexicon

path = datadir("sequence_recognition", "overlap_lexicon")

lexicon = let
    dictionary = getdictionary(["POLLEN", "GOLD", "GOLDEN", "DOLL", "LOP", "GOD", "LOG", "POLL", "GOAL", "DOG"])
    # dictionary = getdictionary(["AB", "CD"])
    duration = getduration(dictionary, 50ms)
    config_lexicon = (ph_duration=duration, dictionary=dictionary)
    lexicon = generate_lexicon(config_lexicon)
end

exp_config = (      # Sequence parameters
                    init_silence=1s, 
                    repetition=200, 
                    silence=1, 
                    peak_rate=8kHz, 
                    start_rate=8kHz, 
                    decay_rate=10ms,
                    proj_strength=20pA,
                    p_post = 0.08f0,
                    targets= [:d],
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
            p_post = exp_config.p_post
            )



## Merge network and stimuli in model
network = get_network(;exp_config...)
stim, seq = SNNUtils.step_input_sequence(network = network, lexicon = lexicon; exp_config..., )
model = merge_models(network, stim)
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
SNN.monitor([model.syn...], [ :W], sr=10Hz)
mytime = SNN.Time()
SNN.train!(model=model, duration= sequence_end(seq), pbar=true, dt=0.125, time=mytime)

savemodel(path=path, name="associative_phase", model=model, info=model_info, lexicon=lexicon, config=exp_config, mytime=mytime, seq=seq)
##

T = get_time(mytime)
Trange = T-1s:1ms:T-100ms
names, pops = filter_populations(model.stim) |> subpopulations
pr1 = SNN.raster(model.pop.E, Trange, populations=pops, names=names)
# pr2 = SNN.raster(model.pop, Trange)
pr2 = plot_activity(model, Trange)
layout = @layout [a{0.3h}; 
                   b{0.7h}
                   ]
plot(pr1, pr2, layout = layout, size = (800, 1400))


## Recall phase
@unpack model, seq, mytime, lexicon, config = load_model(path, "associative_phase", model_info)


recall_config = (;config..., STDP=false, words=false,)
seq = randomize_sequence!(;seq=seq, model=model, recall_config...)
model.syn.E_to_E.param.active[1] = recall_config.STDP

SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
SNN.monitor([model.syn...], [ :W], sr=10Hz)
duration = sequence_end(seq)
mytime = SNN.Time()
SNN.train!(model=model, duration= duration, pbar=true, dt=0.125, time=mytime)

data = (@strdict seq mytime lexicon recall_config) |> dict2ntuple
path = datadir("sequence_recognition", "overlap_lexicon")
save_model(path=path, name="recall_phase", model =model, info=recall_config; data...)
filesize(model_path) |> Base.format_bytes
basename(model_path)