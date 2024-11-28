using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

##
function run_network(params, name)
    @unpack exc, pv, sst, plasticity, connectivity = params
    # Number of neurons in the network
    NE = 1000
    NI = NE ÷ 4
    NI1 = round(Int,NI * 0.35)
    NI2 = round(Int,NI * 0.65)
    # Import models parameters
    # Define interneurons I1 and I2
    @unpack dends, NMDA, param, soma_syn, dend_syn = exc
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
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IsIs...)
    I2_to_E = SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.EdIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)
    # Define normalization
    norm = SNN.SynapseNormalization(NE, [E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    # background noise
    stimuli = Dict(
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
network = run_network(bursty_dendritic_network, "bursty_dendritic_network" )


# Stimulus
dictionary = Dict(:AB=>[:A, :B], :BA=>[:B, :A])
duration = Dict(:A=>50, :B=>50, :_=>50)
config_lexicon = ( ph_duration=duration, dictionary=dictionary)
config_sequence = (init_silence=1s, seq_length=100, silence=1,)
lexicon = generate_lexicon(config_lexicon)

stim, seq = SNNUtils.step_input_sequence(network = network, 
                                    targets= [:d],
                                    lexicon = lexicon, 
                                    config_sequence = config_sequence, 
                                    peak_rate=4kHz, 
                                    start_rate=4kHz, 
                                    decay_rate=10ms,
                                    p_post = 0.15f0)

# Merge network and stimuli in model
model = merge_models(network, stim)

# SNN.monitor(network.pop.E, [:fire, :v_d, :v_s])p
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [ :v_d, :v_s], sr=200Hz)
duration = sequence_end(seq)
mytime = SNN.Time()
SNN.train!(model=model, duration= duration, pbar=true, dt=0.125, time=mytime)
##
T = get_time(mytime)
Trange = T-2s:1ms:T
names, pops = filter_populations(model.stim, "noise") |> subpopulations
pr = SNN.raster(network.pop.E, Trange, populations=pops, names=names)

## Target activation with stimuli
@unpack stim = model
pv = plot()
cells = collect(intersect(Set(stim.AB_d.cells)))
control_cells = rand(1:model.pop.E.N, length(cells))
control_cells = collect(intersect(Set(stim.BA_d.cells)))

SNN.vecplot!(pv, model.pop.E, :v_d, r = Trange, neurons = control_cells, dt = 0.125, pop_average = true, c = :grey, lw=2, ls=:dot)
SNN.vecplot!(pv, model.pop.E, :v_d, r = Trange, neurons = cells, dt = 0.125, pop_average = true, lw=3)
myintervals = sign_intervals(:AB, seq)
vline!(myintervals./1000, c = :black, ls=:dash)
plot!(title = "Depolarization of :AB vs :BA with stimuli AB")
plot!(xlabel = "Time (s)", ylabel = "Membrane potential (mV)")
plot!(margin = 5Plots.mm)
plot(pv, pr, layout = (2, 1), size = (800, 800))
# %%
