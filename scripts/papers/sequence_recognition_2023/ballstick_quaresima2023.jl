using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

##

include("../../parameters/dendritic_network.jl")
function get_network(params, name)
    # @unpack exc, pv, sst, plasticity, connectivity = params
    pv = duarte2019.PV
    sst = duarte2019.SST
    @unpack connectivity, plasticity = quaresima2023
    @unpack dends, NMDA, param, soma_syn, dend_syn = quaresima2022
    # Number of neurons in the network
    NE = 1200
    NI = NE ÷ 4
    NI1 = round(Int,NI * 0.35)
    NI2 = round(Int,NI * 0.65)
    # Import models parameters
    # Define interneurons I1 and I2
    # @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    E = SNN.BallAndStickHet(; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="Exc")
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
    stimuli = Dict(
        :s   => SNN.PoissonStimulus(E,  :he_s,  param=4.0kHz, cells=:ALL, μ=1.f0, name="noise_d",),
        :d   => SNN.PoissonStimulus(E,  :he_d,  param=2.0kHz, cells=:ALL, μ=1.f0, name="noise_d",),
        :i1  => SNN.PoissonStimulus(I1, :ge,   param=1.5kHz, cells=:ALL, μ=1.f0,  name="noise_i1"),
        :i2  => SNN.PoissonStimulus(I2, :ge,   param=2.0kHz, cells=:ALL, μ=1.8f0, name="noise_i2")
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    # Return the network as a model
    merge_models(pop, syn, stimuli, name=name)
end


# Define the network
network = get_network(bursty_dendritic_network, "bursty_dendritic_network" )
dictionary = getdictionary(["POLLEN", "GOLD", "GOLDEN", "DOLL", "LOP", "GOD", "LOG", "POLL", "GOAL", "DOG"])
duration = getduration(dictionary, 50ms)


config_lexicon = ( ph_duration=duration, dictionary=dictionary)
config_sequence = (init_silence=1s, repetition=150, silence=1,)
lexicon = generate_lexicon(config_lexicon)
stim, seq = SNNUtils.step_input_sequence(network = network, 
                                    words=true,
                                    targets= [:d],
                                    lexicon = lexicon, 
                                    peak_rate=8kHz, 
                                    start_rate=8kHz, 
                                    decay_rate=10ms,
                                    p_post = 0.05f0;
                                    config_sequence..., 
                                    )



##
name = DrWatson.savename("associative_phase", model_config, "jld2")
model_path = datadir("sequence_recognition", "overlap_lexicon", name) |> path -> (mkpath(dirname(path)); path)

# Merge network and stimuli in model
model = merge_models(network, stim)
model_config = (vd = -70mV, input_rate=8kHz)
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
SNN.monitor([model.syn...], [ :W], sr=10Hz)
duration = sequence_end(seq)
mytime = SNN.Time()
SNN.train!(model=model, duration= duration, pbar=true, dt=0.125, time=mytime)

DrWatson.save(model_path, @strdict model seq mytime lexicon config_sequence config_lexicon)
filesize(model_path) |> Base.format_bytes
basename(model_path)

## Recall phase
name = DrWatson.savename("recall_phase", model_config, "jld2")
model_path = datadir("sequence_recognition", "overlap_lexicon", name) |> path -> (mkpath(dirname(path)); path)

@unpack model, seq, mytime, lexicon = copy_model(model_path)
seq = randomize_sequence!(;seq=seq, model=model, target=:d, words=false, config_sequence...)
config_sequence = (init_silence=1s, repetition=150, silence=1,)
model.syn.E_to_E.param.active[1]=false
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
SNN.monitor([model.syn...], [ :W], sr=10Hz)
duration = sequence_end(seq)
mytime = SNN.Time()
SNN.train!(model=model, duration= duration, pbar=true, dt=0.125, time=mytime)

DrWatson.save(model_path, @strdict model seq mytime lexicon config_sequence config_lexicon)
filesize(model_path) |> Base.format_bytes
basename(model_path)