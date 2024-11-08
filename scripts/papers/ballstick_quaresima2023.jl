using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

##
function ballstick_network(;
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
    # Import models parameters
    # Define interneurons I1 and I2
    @unpack dends, NMDA, param, soma_syn, dend_syn = E_params
    E = SNN.BallAndStickHet(; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="Exc")
    I1 = SNN.IF(; N = NI1, param = I1_params, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = I2_params, name="I2_sst")
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
        :noise_s   => SNN.PoissonStimulus(E,  :he_s,  param=4.0kHz, cells=:ALL, μ=5.f0, name="noise"),
        :noise_i1  => SNN.PoissonStimulus(I1, :ge,   param=1.8kHz, cells=:ALL, μ=1.f0, name="noise"),
        :noise_i2  => SNN.PoissonStimulus(I2, :ge,   param=2.5kHz, cells=:ALL, μ=1.5f0, name="noise")
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    # Return the network as a model
    merge_models(pop, syn, stimuli)
end


# Sequence input
function step_input_sequence(;network, targets=[:d], lexicon, config_sequence, seed=1234)
    @unpack E = network.pop
    seq = generate_sequence(lexicon, config_sequence, seed)

    function step_input(x, param::PSParam) 
        intervals::Vector{Vector{Float32}} = param.variables[:intervals]
        if time_in_interval(x, intervals)
            my_interval = start_interval(x, intervals)
            return 2kHz * 3kHz*(1-exp(-(x-my_interval)/10))
            # return 0kHz
        else
            return 0kHz
        end
    end

    all_symbols = collect(Iterators.flatten(seq.symbols))
    stim = Dict{Symbol,Any}()
    for s in seq.symbols.words
        param = PSParam(rate=step_input, variables=Dict(:intervals=>sign_intervals(s, seq)))
        for t in targets
            push!(stim,s  => SNN.PoissonStimulus(E, :he, t, μ=4.f0, param=param, name="w_$s"))
        end
    end
    for s in seq.symbols.phonemes
        param = PSParam(rate=step_input, variables=Dict(:intervals=>sign_intervals(s, seq)))
        for t in targets
            push!(stim,s  => SNN.PoissonStimulus(E, :he, t, μ=4.f0, param=param, name="$s"))
        end
    end
    stim = dict2ntuple(stim)
    stim, seq
end

# %% [markdown]
# # Quaresima et al. 2023
# %%

# Define the network
I1_params = duarte2019.PV
I2_params = duarte2019.SST
E_params = quaresima2022
@unpack connectivity, plasticity = quaresima2023
network = ballstick_network(I1_params=I1_params, I2_params=I2_params, E_params=E_params, connectivity=connectivity, plasticity=plasticity)
# Stimulus
dictionary = Dict(:AB=>[:A, :B], :BA=>[:B, :A])
duration = Dict(:A=>50, :B=>50, :_=>200)
config_lexicon = ( ph_duration=duration, dictionary=dictionary)
config_sequence = (init_silence=1s, seq_length=200, silence=1,)
lexicon = generate_lexicon(config_lexicon)
stim, seq = step_input_sequence(network=network, lexicon=lexicon, config_sequence=config_sequence, seed=1234)
model = merge_models(network, stim)

# %%
SNN.monitor(network.pop.E, [:fire, :v_d])
SNN.monitor([network.pop...], [:fire])
duration = sequence_end(seq)
mytime = SNN.Time()
SNN.train!(model=model, duration= 15s, pbar=true, dt=0.125, time=mytime)
##
SNN.raster(model.pop, (10s,15s))
names, pops = subpopulations(stim)
stim
pr = SNN.raster(network.pop.E, (10s,15s), populations=pops, names=names)

## Target activation with stimuli
pv=plot()
cells = collect(intersect(Set(stim.AB.cells)))
SNN.vecplot!(pv,model.pop.E, :v_d, r=5s:7.0s, neurons=cells, dt=0.125, pop_average=false)
myintervals = sign_intervals(:AB, seq)
vline!(myintervals, c=:red)
plot!(title="Depolarization of :A with stimuli :w_AB")
plot!(xlabel="Time (s)", ylabel="Membrane potential (mV)")
plot!(margin=5Plots.mm)
plot(pv, pr, layout=(2,1), size=(800, 600))