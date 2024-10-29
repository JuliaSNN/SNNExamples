using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

NE = 1000
NI = 1000 ÷ 4
NI1 = round(Int,NI * 0.35)
NI2 = round(Int,NI * 0.65)

# Import models parameters
I1_params = duarte2019.PV
I2_params = duarte2019.SST
E_params = quaresima2022
@unpack connectivity, plasticity = quaresima2023

network = let
    # Number of neurons in the network
    # Define neurons and synapses in the network
    @unpack dends, NMDA, param, soma_syn, dend_syn = E_params
    E = SNN.BallAndStick((150um, 400um);
        N = NE,
        soma_syn = soma_syn,
        dend_syn = dend_syn,
        NMDA = NMDA,
        param = param
    )

    # Define interneurons I1 and I2
    I1 = SNN.IF(; N = NI1, param = I1_params)
    I2 = SNN.IF(; N = NI2, param = I2_params)
    # Define synaptic interactions between neurons and interneurons

    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...,)

    I1_to_E = SNN.CompartmentSynapse(
        I1,
        E,
        :s,
        :hi;
        param = plasticity.iSTDP_rate,
        connectivity.EIf...,
    )

    I2_to_E = SNN.CompartmentSynapse( # what learning rate should I use?
        I2,
        E,
        :d,
        :hi;
        param = plasticity.iSTDP_potential,
        connectivity.EdIs...,
    )

    # Define recurrent connections in the network
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...)
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IsIs...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)

    E_to_E = SNN.CompartmentSynapse(
        E,
        E,
        :d,
        :he;
        param = plasticity.vstdp,
        connectivity.EdE...,
    )

    # Define normalization
    norm = SNN.SynapseNormalization(NE, [E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))

    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    
	# Return the network as a tuple
    (pop = pop, syn = syn)
end

## Stimulus
# Background noise
stimuli = Dict(
    :noise_s   => SNN.PoissonStimulus(network.pop.E,  :he_s,  param=6.0kHz, cells=:ALL, μ=2.7f0,),
    :noise_i1  => SNN.PoissonStimulus(network.pop.I1, :ge,   param=2.0kHz, cells=:ALL, μ=1.f0),
    :noise_i2  => SNN.PoissonStimulus(network.pop.I2, :ge,   param=2.4kHz, cells=:ALL, μ=2.f0),
)
baseline = merge_models(stimuli, network)


# Sequence input
dictionary = Dict(:AB=>[:A, :B], :BA=>[:B, :A], :AC=>[:A, :C], :CA=>[:C, :A], :BC=>[:B, :C], :CB=>[:C, :B])
duration = Dict(:A=>50, :B=>50, :_=>200, :C=>50)

config_lexicon = ( ph_duration=duration, dictionary=dictionary)
lexicon = generate_lexicon(config_lexicon)

config_sequence = (init_silence=1s, seq_length=500, silence=1,)
seq = generate_sequence(lexicon, config_sequence, 1234)

function step_input(x, param::PSParam) 
    intervals::Vector{Vector{Float32}} = param.variables[:intervals]
    if time_in_interval(x, intervals)
        return 10kHz
    else
        return 0kHz
    end
end

stim = Dict{Symbol,Any}()
for p in seq.symbols.phonemes
    param = PSParam(rate=step_input, 
                    variables=Dict(:intervals=>sign_intervals(p, seq)))
    push!(stim,p  => SNN.PoissonStimulus(network.pop.E, :he, :d, μ=15.f0, param=param, p_post=0.1f0))
end


ext = dict2ntuple(stim)
model = merge_models(baseline, ext=stim)


# %%
SNN.monitor(model.pop.E, [:fire, :v_d])
SNN.monitor([network.pop...], [:fire])
duration = sequence_end(seq)
# @profview 
mytime = SNN.Time()
SNN.train!(model=model, duration= 5s, pbar=true, dt=0.125, time=mytime)
SNN.raster([model.pop...], (000,5000))
SNN.raster([network.pop.E], (4000,5000), populations=[stim[i].cells for i in keys(stim)])



## Target activation with stimuli
p = plot()
cells = collect(intersect(Set(ext.A.cells)))
SNN.vecplot!(p,model.pop.E, :v_d, r=3.5s:4.5s, neurons=cells, dt=0.125, pop_average=true)
myintervals = sign_intervals(:A, seq)
vline!(myintervals, c=:red)
##
model.syn.I2_to_E.W


pointer(model.pop.I1.name)
pointer(model.pop.E.name)
