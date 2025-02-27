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
using StatsPlots
using Logging

# %%
# Instantiate a  Symmetric STDP model with these parameters:
include("parameters.jl")
include("protocol.jl")

path_load = datadir("Lagzi2022_AssemblyFormation", "mixed_inh", "InputFluctuations")
path_response = datadir("Lagzi2022_AssemblyFormation", "mixed_inh", "SoundResponseExp2")
@unpack rates, interval= load(datadir("ExpData", "ACrates.jld2")) |> dict2ntuple
sound = mean(rates)

models = []
Threads.@threads for t in eachindex(NSSTs)
        @unpack stim_τ, stim_rate = config
        info = (τ= stim_τ, rate=stim_rate, signal=:off, NSST=NSSTs[t])
        !isfile(save_name(path=path_load, name="Model_sst", info=info)) && continue
        data = load_model(path_load, "Model_sst", info)
        model = data.model
        network_config = data.config

        # Set lower inhibitory noise
        model.stim.inh_noise.param.rate .=0.8
        model.syn.E1_to_E1.W .*= 1/mean(model.syn.E1_to_E1.W)
        model.syn.E2_to_E2.W .*= 1/mean(model.syn.E1_to_E1.W)
        model.syn.E1_to_E2.W .*= 0.5/mean(model.syn.E1_to_E1.W)
        model.syn.E2_to_E1.W .*= 0.5/mean(model.syn.E1_to_E1.W)

        # Remove external signal
        no_ext_input = filter_items(model.stim,condition= x-> !occursin("ExtSignal",x.name))
       
        # Set model, stimulus and exp parameters
        exp_config = (delay=1s, repetitions=40, warmup=20s, target_pop =:E1 , input_strength=200)

        model = merge_models(model.pop, no_ext_input, model.syn, silent=true)
        sound_stim = deepcopy(SNN.sample_inputs(exp_config.input_strength, sound, interval))
        rec_interval = 0:3s

        # Set model info
        info = (τ= stim_τ, rate=stim_rate, signal=:off, NSST=NSSTs[t], plasticity=false,)
        #------------------------------------------------------------------------------
        # Test sound response without plasticity and save model
        clear_records(model)
        TTL, sim_interval = test_sound_response(model, sound_stim, plasticity=false; exp_config...)
        save_model(;path=path_response, model=model, name="SoundResponse", info=info, exp_config=exp_config, sound=sound_stim, TTL=TTL, sim_interval=sim_interval)
        # Record sound response and save recordings
        recordings = record_sound_response(model; TTL, sim_interval, rec_interval)
        rec_name = joinpath(path_response, savename("SoundResponseRecordings", info, "jld2"))
        save(rec_name, @strdict recordings=recordings info=info rec_interval=rec_interval)

        info = (τ= stim_τ, rate=stim_rate, signal=:off, NSST=NSSTs[t], plasticity=true)
        #------------------------------------------------------------------------------
        clear_records(model)
        TTL, sim_interval = test_sound_response(model, sound_stim, plasticity=true; exp_config...)
        save_model(;path=path_response, model=model, name="SoundResponse", info=info, exp_config=exp_config, network_config=config, sound=sound_stim, TTL=TTL, sim_interval=sim_interval)
        recordings = record_sound_response(model; TTL, sim_interval, rec_interval)
        rec_name = joinpath(path_response, savename("SoundResponseRecordings", info, "jld2"))
        save(rec_name, @strdict recordings=recordings info=info rec_interval=rec_interval)
end


# ##
# sound_stim = deepcopy(SNN.sample_inputs(200, sound, interval))
# model = models[end-5]
# model.pop.SST1.N
# model.stim.inh_noise.param.rate .=0.5
# exp_config = (delay=1s, repetitions=1, warmup=10s, inter_trial_interval=5s, target_pop =:E1 )
# stimulus_TTL, sim_interval = test_sound_response(model, sound_stim, plasticity=false; exp_config...)
# raster(model.pop, 0.1s:4s)
# recordings = record_sound_response(model, sim_interval, exp_config.rec_interval)
# frs, r, names = firing_rate(model.pop; interval=0:10:15s, τ=10ms)
# # rec_name = joinpath(path, savename("SoundResponseRecordings", info, ".jld2"))
# # save(rec_name, @strdict recordings=recordings info=info rec_interval=exp_config.rec_interval)
# # recs = load(rec_name)["recordings"]

# ##

# plot(rec_interval,mean(recordings[:,:,1], dims=2)[:,1], label="E1")
# plot!(rec_interval,mean(recordings[:,:,2], dims=2)[:,1], label="E2")
# plot!(rec_interval,mean(recordings[:,:,3], dims=2)[:,1], label="PV")
# plot!(rec_interval,mean(recordings[:,:,4], dims=2)[:,1], label="SST1")
# ##

# # stimulus_TTL
# ##
# # T = get_time(mytime)
# #     shift_spikes!(stim, 1s) 
# #     sim!(model=model, duration=3s, time=mytime)


# # W, r = record(model.syn.PV_to_E1, :W) 
# # W =mean(W, dims=1)[1,:]
# # plot(r, W)
# # #   |> plot
