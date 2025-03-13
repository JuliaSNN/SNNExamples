using Revise
using DrWatson
SNN.@load_units;
using Plots
using SNNUtils
using SpikingNeuralNetworks
using Statistics
using Random
using StatsBase
using SparseArrays
using Distributions

include("models.jl")
include("parameters.jl")
# Instantiate a  Symmetric STDP model with these parameters:

##
@unpack stim_rate, stim_τ = config 
local_config = (; config...,
    stim_τ = stim_τ,
    adex_param = AdExSynapseParameter(a=0, b=0),
    stim_rate = stim_rate,
    I_noise = 0.8ext_rate,
    E_noise = 0.5ext_rate,
    signal_param = Dict(:X => 2.0f0,
        :σ => 0.4kHz,
        :dt => 0.125f0,
        :θ => 1/stim_τ,
        :μ => stim_rate*ext_rate
        ),
)
info = (τ= local_config.stim_τ, rate=local_config.stim_rate)
model = network(local_config=local_config, type= :sst)
# train!(model=model, duration=20s, pbar=true)
# save_model(path=path,name="Model_sst", model=model, info=info, config=config)
SNN.monitor(model.pop.E1, [(:v,1:20)], sr=100Hz)
SNN.monitor(model.pop.E1, [(:g,1:20)], sr=100Hz)
train!(model=model, duration=5s, pbar=true)
p1 = SNN.vecplot(model.pop.E1, :g, sym_id=1, neurons=1:20, interval=0s:4s, pop_average=true)
raster(model.pop, 1s:4s)
model_nmda = deepcopy(model)
fr, r = firing_rate(model_nmda.pop.E1, interval=1s:4s, pop_average=true)

##
@unpack stim_rate, stim_τ = config 
local_config = (; config...,
    stim_τ = stim_τ,
    adex_param = AdExParameter(a=0, b=0),
    stim_rate = stim_rate,
    I_noise = 0.8ext_rate,
    E_noise = 0.5ext_rate,
    signal_param = Dict(:X => 2.0f0,
        :σ => 0.4kHz,
        :dt => 0.125f0,
        :θ => 1/stim_τ,
        :μ => stim_rate*ext_rate
        ),
)
info = (τ= local_config.stim_τ, rate=local_config.stim_rate)
model = network(local_config=local_config, type= :sst)
# train!(model=model, duration=20s, pbar=true)
# save_model(path=path,name="Model_sst", model=model, info=info, config=config)
SNN.monitor(model.pop.E1, [(:v,1:20)], sr=100Hz)
SNN.monitor(model.pop.E1, [(:ge,1:20)], sr=100Hz)
train!(model=model, duration=5s, pbar=true)
p2 = SNN.vecplot(model.pop.E1, :ge, sym_id=1, neurons=1:20, interval=0s:4s, pop_average=true)
raster(model.pop, 1s:4s)
# plot(p1,p2, layout=(1,2), size=(800, 800), link=:y)
fr, r = firing_rate(model.pop.E1, interval=1s:4s, pop_average=true)



# path = datadir("Lagzi2022_AssemblyFormation")
# Threads.@threads for t in eachindex(τs)
#     for r in eachindex(stim_rates)
#         stim_rate = stim_rates[r]
#         stim_τ = τs[t]
#         info = (τ= stim_τ, rate=stim_rate)
#         model = load_model(path, "Model_sst", info).model

#         info = (τ= stim_τ, rate=stim_rate, signal=:off)
#         model.stim.exc_noise1.param.rate.=2kHz
#         model.stim.exc_noise2.param.rate.=2kHz
#         model.stim.signal_signal_E1.param.active[1] = false
#         model.stim.signal_signal_E2.param.active[1] = false
#         SNN.monitor(model.pop, [:fire])

#         SNN.monitor(model.syn.E1_to_E1, [:W], sr=10Hz)
#         SNN.monitor(model.syn.E1_to_E2, [:W], sr=10Hz)
#         SNN.monitor(model.syn.E2_to_E1, [:W], sr=10Hz)
#         SNN.monitor(model.syn.E2_to_E2, [:W], sr=10Hz)

#         SNN.monitor(model.syn.SST1_to_E1, [:W], sr=10Hz)
#         SNN.monitor(model.syn.SST1_to_E2, [:W], sr=10Hz)
#         SNN.monitor(model.syn.SST2_to_E1, [:W], sr=10Hz)
#         SNN.monitor(model.syn.SST2_to_E2, [:W], sr=10Hz)

#         SNN.monitor(model.syn.PV_to_E1, [:W], sr=10Hz)
#         SNN.monitor(model.syn.PV_to_E2, [:W], sr=10Hz)

#         no_ext_info = (τ= stim_τ, rate=stim_rate, signal=:off)
#         isfile(get_path(path=path, name="Model_sst", info=no_ext_info)) && continue
#         train!(model=model, duration=500s, pbar=true)
#         save_model(path=path,name="Model_sst", model=model, info=no_ext_info, config=config)
#     end
# end 