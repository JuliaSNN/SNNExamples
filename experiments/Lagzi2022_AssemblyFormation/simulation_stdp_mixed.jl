using Revise
using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Statistics
using Random
using StatsBase
using SparseArrays
using Distributions

include("models.jl")
include("parameters.jl")
##
# Instantiate a  Symmetric STDP model with these parameters:
path = datadir("Lagzi2022_AssemblyFormation/mixed_inh", "LowNoise") |> mkpath
NSSTs

Threads.@threads for t in eachindex(NSSTs)
    @unpack stim_τ, stim_rate = config
    local_config = (; config...,
        I_noise = 0.8,
        E_noise = 0.8,
        signal_param = Dict(:X => 2.0f0,
            :σ => 0.4kHz,
            :dt => 0.125f0,
            :θ => 1/stim_τ,
            :μ => stim_rate*ext_rate
            ),
            NSST = NSSTs[t]
    )
    info = (NSST = NSSTs[t], τ= local_config.stim_τ, rate=local_config.stim_rate)
    @show save_name(path=path, name="Model_sst", info=info)
    isfile(save_name(path=path, name="Model_sst", info=info)) && continue
    model = network(local_config=local_config, type= :sst)
    train!(model=model, duration=500s, pbar=true)
    save_model(path=path,name="Model_sst", model=model, info=info, config=local_config)
end


Threads.@threads for t in eachindex(NSSTs)
    @unpack stim_τ, stim_rate = config
    info = (NSST = NSSTs[t], τ= config.stim_τ, rate=config.stim_rate)
    model = load_model(path, "Model_sst", info).model

    info = (NSST = NSSTs[t], τ= config.stim_τ, rate=config.stim_rate, signal=:off)
    model.stim.exc_noise1.param.rate.=1.2kHz
    model.stim.exc_noise2.param.rate.=1.2kHz
    model.stim.signal_signal_E1.param.active[1] = false
    model.stim.signal_signal_E2.param.active[1] = false
    SNN.monitor(model.pop, [:fire])

    SNN.monitor(model.syn.E1_to_E1, [:W], sr=10Hz)
    SNN.monitor(model.syn.E1_to_E2, [:W], sr=10Hz)
    SNN.monitor(model.syn.E2_to_E1, [:W], sr=10Hz)
    SNN.monitor(model.syn.E2_to_E2, [:W], sr=10Hz)

    SNN.monitor(model.syn.SST1_to_E1, [:W], sr=10Hz)
    SNN.monitor(model.syn.SST1_to_E2, [:W], sr=10Hz)
    SNN.monitor(model.syn.SST2_to_E1, [:W], sr=10Hz)
    SNN.monitor(model.syn.SST2_to_E2, [:W], sr=10Hz)

    SNN.monitor(model.syn.PV_to_E1, [:W], sr=10Hz)
    SNN.monitor(model.syn.PV_to_E2, [:W], sr=10Hz)

    no_ext_info = (NSST = NSSTs[t], τ= config.stim_τ, rate=config.stim_rate, signal=:off)
    model.stim.exc_noise1.param.rate.=2kHz
    isfile(save_name(path=path, name="Model_sst", info=no_ext_info)) && continue
    train!(model=model, duration=500s, pbar=true)
    save_model(path=path,name="Model_sst", model=model, info=no_ext_info, config=config)
end 