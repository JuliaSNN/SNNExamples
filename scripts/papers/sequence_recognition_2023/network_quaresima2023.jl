using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

## Define the network parameters

# Import models parameters

function tripod_network(;
            I1_params, 
            I2_params, 
            E_params, 
            connectivity,
            plasticity)
    # Number of neurons in the network
    NE = 1000
    NI = NE ÷ 4
    NI1 = round(Int,NI * 0.35)
    NI2 = round(Int,NI * 0.65)

    # Define interneurons I1 and I2
    @unpack dends, NMDA, param, soma_syn, dend_syn = E_params
    E = SNN.Tripod(dends...; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param)
    I1 = SNN.IF(; N = NI1, param = I1_params, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = I2_params, name="I2_sst")

    # Define synaptic interactions between neurons and interneurons
    E_to_E1_d1 = SNN.CompartmentSynapse(E, E, :d1, :he; connectivity.EdE...)
    E_to_E2_d2 = SNN.CompartmentSynapse(E, E, :d2, :he; connectivity.EdE...)
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...)
    I1_to_E = SNN.CompartmentSynapse(I1, E, :s, :hi; param = plasticity.iSTDP_rate, connectivity.EIf...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...)
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IsIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)
    I2_to_E_d1 = SNN.CompartmentSynapse(I2, E, :d1, :hi; param = plasticity.iSTDP_potential, connectivity.EdIs...)
    I2_to_E_d2 = SNN.CompartmentSynapse(I2, E, :d2, :hi; param = plasticity.iSTDP_potential, connectivity.EdIs...)

    # Define normalization
    recurrent_norm = [
        SNN.SynapseNormalization(NE, [E_to_E1_d1], param = SNN.MultiplicativeNorm(τ = 20ms)),
        SNN.SynapseNormalization(NE, [E_to_E2_d2], param = SNN.MultiplicativeNorm(τ = 20ms))
    ]

    # background noise
    stimuli = Dict(
        :d1   => SNN.PoissonStimulus(E,  :he_d1,  param=2.0kHz, cells=:ALL, μ=1.f0, name="noise_d1",),
        :d2   => SNN.PoissonStimulus(E,  :he_d2,  param=2.0kHz, cells=:ALL, μ=1.f0, name="noise_d2",),
        :i1  => SNN.PoissonStimulus(I1, :ge,   param=1.5kHz, cells=:ALL, μ=1.f0,  name="noise_i1"),
        :i2  => SNN.PoissonStimulus(I2, :ge,   param=2.0kHz, cells=:ALL, μ=1.8f0, name="noise_i2")
    )

    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E_d1 I2_to_E_d2 I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E1_d1 E_to_E2_d2 norm1=recurrent_norm[1] norm2=recurrent_norm[2])

    # Return the network as a model
    merge_models(pop, syn, stimuli)
end

# Define the network
I1_params = duarte2019.PV
I2_params = duarte2019.SST
E_params = quaresima2022
@unpack connectivity, plasticity = quaresima2023
network = tripod_network(I1_params=I1_params, I2_params=I2_params, E_params=E_params, connectivity=connectivity, plasticity=plasticity)

# Stimulus
dictionary = Dict(:AB=>[:A, :B], :BA=>[:B, :A])
duration = Dict(:A=>50, :B=>50, :_=>50)
config_lexicon = ( ph_duration=duration, dictionary=dictionary)
config_sequence = (init_silence=1s, seq_length=100, silence=1,)
lexicon = generate_lexicon(config_lexicon)
stim, seq = SNNUtils.step_input_sequence(network = network, 
                                    targets= [:d1, :d2],
                                    lexicon = lexicon, 
                                    config_sequence = config_sequence, 
                                    peak_rate=4kHz, 
                                    start_rate=4kHz, 
                                    decay_rate=10ms,
                                    p_post = 0.2f0)

model = merge_models(network, stim)

model_nostim = deepcopy(merge_models(network))

# %%
# Run the model
# %%
SNN.monitor(model.pop.E, [:fire, :v_d1,])
SNN.monitor([network.pop...], [:fire])
duration = sequence_end(seq)
# @profview 
mytime = SNN.Time()
SNN.train!(model=model, duration= 10s, pbar=true, dt=0.125, time=mytime)

# %%
# Analyse the results
# %%

names, pops = subpopulations(stim.d1)
pr = SNN.raster(model.pop.E, (5s,10s), populations=pops, names=names)
pr = SNN.raster(model.pop, (8s,10s))

## Target activation with stimuli
pv=plot()
cells = collect(intersect(Set(stim.d1.A.cells)))
SNN.vecplot!(pv,model.pop.E, :v_d1, r=3s:5s, neurons=cells, dt=0.125, pop_average=true)
myintervals = sign_intervals(:A, sequence)
vline!(myintervals, c=:red)
plot!(title="Depolarization of :A with stimuli :w_AB")
plot!(xlabel="Time (s)", ylabel="Membrane potential (mV)")
plot!(margin=5Plots.mm)
##
plot(pv, pr, layout=(2,1), size=(800, 600))

histogram(network.syn.I2_to_E_d2.W)