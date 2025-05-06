using DrWatson
using Revise
findproject() |> quickactivate

using SpikingNeuralNetworks
using Plots
using Distributions
using SNNUtils
SNN.@load_units
##

data_path = datadir("Vlachos2025_Sequences") |> mkpath

include("parameters.jl")
include("model.jl")

merge_models(name="Vlachos2025_Sequences")

exc_params = SNN.AdExMultiTimescaleParameter(
    τm = 20ms, # Assuming τm = C / gL
    Vt = -52mV, # Convert to mV
    Vr = -60mV, # Convert to mV
    El = -70mV, # Convert to mV
    R = 1 / 15nS, # Assuming R = 1 / gL
    ΔT = 2mV, # Convert to mV
    τw = 150ms, # Convert to ms
    a = 4nS, # Convert to nS
    b = 0pA, # Convert to pA
    τabs = 1ms, # Convert to ms
    E_i = -75mV, # Convert to mV
    E_e = 0mV, # Convert to mV
    At = 10mV, # Convert to mV
    τt = 30ms, # Convert to ms
    # Synaptic timescales
    τr = [1ms, 0.5ms, 5ms],# Default value
    τd = [6ms, 2ms, 20ms], # Default value
    exc_receptors = [1],
    inh_receptors = [2,3]
)

inh_params = SNN.AdExMultiTimescaleParameter(
    τm = 20ms, # Assuming τm = C / gL
    Vt = -52mV, # Convert to mV
    Vr = -60mV, # Convert to mV
    El = -70mV, # Convert to mV
    R = 1 / 15nS, # Assuming R = 1 / gL
    ΔT = 2mV, # Convert to mV
    τw = 150ms, # Convert to ms
    a = 4nS, # Convert to nS
    b = 0pA, # Convert to pA
    τabs = 1ms, # Convert to ms
    E_i = -75mV, # Convert to mV
    E_e = 0mV, # Convert to mV
    At = 10mV, # Convert to mV
    τt = 30ms, # Convert to ms
    # Synaptic timescales
    τr = [1ms, 0.5ms, 5ms],# Default value
    τd = [6ms, 2ms, 20ms], # Default value
    exc_receptors = [1],
    inh_receptors = [2,3]
)

syn_params = (
    E_to_E = (μ=2.86pF, p=0.2),
    E_to_I1 = (μ=1.27pF, p=0.2),
    E_to_I2 = (μ=1.27pF, p=0.2),

    I1_to_E = (μ=48.7pF, p=0.2),
    I1_to_I1 = (μ=16.2pF, p=0.2),
    I1_to_I2 = (μ=24.3pF, p=0.2),

    I2_to_E = (μ=48pF, p=0.2),
    I2_to_I1 = (μ=16.2pF, p=0.2),
    I2_to_I2 = (μ=32.4pF, p=0.2),
)

##
# ##

##
Exc = AdExMultiTimescale(500, param=exc_params, name="Exc")
Inh1 = AdExMultiTimescale(100, param=inh_params, name="Inh")
Inh2 = AdExMultiTimescale(100, param=inh_params, name="Inh")

populations = (;Exc, Inh1, Inh2)

synapses = (
    E_to_E = SpikingSynapse(Exc, Exc, :h, 1; syn_params.E_to_E...),
    E_to_I1 = SpikingSynapse(Exc, Inh1, :h, 1; syn_params.E_to_I1...),
    E_to_I2 = SpikingSynapse(Exc, Inh2, :h, 1; syn_params.E_to_I2...),
    I1_to_E = SpikingSynapse(Inh1, Exc, :h, 1; syn_params.I1_to_E...),
    I1_to_I1 = SpikingSynapse(Inh1, Inh1, :h, 1; syn_params.I1_to_I1...),
    I1_to_I2 = SpikingSynapse(Inh1, Inh2, :h, 1; syn_params.I1_to_I2...),
    I2_to_E = SpikingSynapse(Inh2, Exc, :h, 1; syn_params.I2_to_E...),
    I2_to_I1 = SpikingSynapse(Inh2, Inh1, :h, 1; syn_params.I2_to_I1...),
    I2_to_I2 = SpikingSynapse(Inh2, Inh2, :h, 1; syn_params.I2_to_I2...),
)

exc_stim_param = CurrentNoiseParameter(Exc.N, I_base =0pA, I_dist=Normal(300pA, 300pA))
exc_stim = CurrentStimulus(Exc, param=stim_param, name="Exc_stim")
inh_stim_param = CurrentNoiseParameter(Inh.N, I_base =0pA, I_dist=Normal(300pA, 300pA))
inh1_stim = CurrentStimulus(Inh, param=stim_param, name="Exc_stim")
inh2_stim = CurrentStimulus(Inh, param=stim_param, name="Exc_stim")


stim = (; exc_stim, inh1_stim, inh2_stim)

stim=(exc=exc_stim, inh=inh_stim)
model = merge_models(;synapses..., stim..., populations..., name= "Vlachos2025_Sequences")

SNN.monitor!(model.pop, [:fire])
SNN.monitor!(model.pop, [:v], sr=8000Hz)
SNN.monitor!(model.pop, [:g], sr=800Hz)

sim!(;model, duration=30s, pbar = true, dt=0.125)

vecplot(model.pop.Exc, :v, interval=2s:0.125:20s, sym_id=3,neurons=1)
vecplot(model.pop.Inh, :v, interval=2s:0.125:20s, sym_id=3,neurons=1)
# vecplot(model.pop.Exc, :g, interval=0:0.5:3s, sym_id=3,neurons=1)

# model.pop.Exc.h
# spiketimes(model.pop.Exc) 