using DrWatson
@quickactivate "network_models"
using Plots
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Random, Statistics, StatsBase
using Statistics, SparseArrays
using StatsPlots
using ProgressBars

# %% [markdown]
# Network


dend_syn = Synapse(EyalGluDend, MilesGabaDend)
dend_syn.AMPA.g0

Random.seed!(123)
network = let
    NE = 400
    NI = 100
    E = SNN.TripodHet(
        N = NE,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
        dend_syn = Synapse(EyalGluDend, MilesGabaDend),
        NMDA = SNN.EyalNMDA,
        param = SNN.AdExSoma(Vr = -50),
    )
    I1 = SNN.IF(; N = NI ÷ 2, param = SNN.IFParameter(τm = 7ms, El = -55mV))
    I2 = SNN.IF(; N = NI ÷ 2, param = SNN.IFParameter(τm = 20ms, El = -55mV))
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge, p = 0.2, σ = 15.0)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge, p = 0.2, σ = 15.0)
    I2_to_E = SNN.CompartmentSynapse(
        I2,
        E,
        :d1,
        :inh,
        p = 0.2,
        σ = 5.0,
        param = SNN.iSTDPParameterPotential(v0 = -50mV),
    )
    I1_to_E = SNN.CompartmentSynapse(
        I1,
        E,
        :s,
        :inh,
        p = 0.2,
        σ = 5.0,
        param = SNN.iSTDPParameterRate(r = 10Hz),
    )
    E_to_E_d1 = SNN.CompartmentSynapse(
        E,
        E,
        :d1,
        :exc,
        p = 0.2,
        σ = 30,
        param = SNN.vSTDPParameter(),
    )
    E_to_E_d2 = SNN.CompartmentSynapse(
        E,
        E,
        :d2,
        :exc,
        p = 0.2,
        σ = 30,
        param = SNN.vSTDPParameter(),
    )
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_E_d1 E_to_E_d2 I1_to_E I2_to_E E_to_I1 E_to_I2)
    recurrent_norm_d1 = SNN.SynapseNormalization(
        E,
        [E_to_E_d1],
        param = SNN.MultiplicativeNorm(τ = 100ms),
    )
    recurrent_norm_d2 = SNN.SynapseNormalization(
        E,
        [E_to_E_d2],
        param = SNN.MultiplicativeNorm(τ = 100ms),
    )
    norm = dict2ntuple(@strdict d1 = recurrent_norm_d1 d2 = recurrent_norm_d2)
    (pop = pop, syn = syn, norm = norm)
end

# background
noise = TripodExcNoise(network.pop.E)

populations = [network.pop..., noise.pop...]
synapses = [network.syn..., noise.syn..., network.norm...]

# populations, synapses = SNN.model([network, noise])
# populations

##
SNN.clear_records([network.pop...])
SNN.train!(populations, synapses, duration = 5000ms, pbar = true, dt = 0.125)


# using BenchmarkTools
# @btime SNN.sim!(populations, synapses, duration = 1000ms)
# @profview SNN.sim!(populations, synapses, duration = 1000ms)

# SNN.raster([network.pop...])
# savefig(plotsdir("example_raster.pdf"))

##
SNN.monitor(network.pop.E, [:v_d1, :v_s, :fire])
SNN.monitor(network.pop.I1, [:fire])
SNN.monitor(network.pop.I2, [:fire])
SNN.sim!(populations, synapses, duration = 1000ms)

SNN.raster([network.pop...])
WI1 = network.syn.I1_to_E.W
WI2 = network.syn.I2_to_E.W
WEd1 = network.syn.E_to_E_d1.W
WEd2 = network.syn.E_to_E_d2.W
spikes = SNN.spiketimes(network.pop.E)
data_new = dict2ntuple(@strdict WI1 WI2 WEd2 WEd1 spikes)
# tagsave(datadir("network_test", "TripodNetwork.jld2"), data, safe=true)
##
data_old = dict2ntuple(DrWatson.load(datadir("network_test", "TripodNetwork.jld2")))
@assert sum(data_new.WI1 .- data_old.WI1) ≈ 0
@assert sum(data_new.WI2 .- data_old.WI2) ≈ 0
@assert sum(data_new.WEd2 .- data_old.WEd2) ≈ 0
@assert sum(data_new.WEd1 .- data_old.WEd1) ≈ 0
