using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

## Define the network parameters
# ! Make them adaptive with these parameters (and verify the others)
PVDuarte = SNN.IFParameter(
    τm = 104.52pF / 9.75nS,
    El = -64.33mV,
    Vt = -38.97mV,
    Vr = -57.47mV,
    τabs = 0.5ms,
    # τe = 0.70ms,
    # τi = 2.50ms
    #     b = 10.0,       #(pA) 'sra' current increment
    #     τw = 144,        #(s) adaptation time constant (~Ca-activated K current inactivation)
#     τabs = 0.52,       #(ms)
)

SSTDuarte = SNN.IFParameter(
    τm = 102.86pF / 4.61nS,
    El = -61mV,
    Vt = -34.4mV,
    Vr = -47.11mV,
    τabs = 1.3ms,
    τre = 0.18ms,
    τde = 1.80ms,
    τri = 0.19ms,
    τdi = 5.00ms,
#     b = 80.5,       #(pA) 'sra' current increment
#     τw = 144,        #(s) adaptation time constant (~Ca-activated K current inactivation)
#     τabs = 1.34,       #(ms)
)

ConnectivityParams = (
    EdE = (p = 0.2,  μ = 10.8, dist = Normal, σ = 1),
    IfE = (p = 0.2,  μ = log(3.7),  dist = LogNormal, σ = 0.1),
    IsE = (p = 0.2,  μ = log(3.7),  dist = LogNormal, σ = 0.1),
    EIf = (p = 0.2,  μ = log(10.8), dist = LogNormal, σ = 0),
    IsIf = (p = 0.2, μ = log(1.4),  dist = LogNormal, σ = 0.25),
    IfIf = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.14),
    EdIs = (p = 0.4, μ = log(10.8), dist = LogNormal, σ = 0),
    IfIs = (p = 0.2, μ = log(0.83), dist = LogNormal, σ = 0.),
    IsIs = (p = 0.2, μ = log(0.83), dist = LogNormal, σ = 0.),

)
    
plasticity_quaresima2023 = (
        iSTDP_rate = SNN.iSTDPParameterRate(η = 1., τy = 5ms, r=5Hz, Wmax = 273.4pF, Wmin = 0.1pF), 
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.1, v0 = -70mV, τy = 5ms, Wmax = 273.4pF, Wmin = 0.1pF),        
        vstdp = SNN.vSTDPParameter(
                A_LTD = 4.0f-5,  #ltd strength          # made 10 times slower
                A_LTP = 14.0f-5, #ltp strength
                θ_LTD = -40.0,  #ltd voltage threshold # set higher
                θ_LTP = -20.0,  #ltp voltage threshold
                τu = 15.0,  #timescale for u variable
                τv = 45.0,  #timescale for v variable
                τx = 20.0,  #timescale for x variable
                Wmin = 1.78,  #minimum ee strength
                Wmax = 41.4,   #maximum ee strength
            )
    )

NE = 1000
NI = 1000 ÷ 4
NI1 = round(Int,NI * 0.35)
NI2 = round(Int,NI * 0.65)
neurons_quaresima2023 = (
    TripodParam= (
        dend  =  [(150um, 400um), (150um, 400um)],
        N = NE,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma), # defines glutamaterbic and gabaergic receptors in the soma
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        NMDA = SNN.EyalNMDA, # NMDA synapse
        param = SNN.AdExSoma(Vr = -55mV, Vt = -50mV),
    ),
    OtherNeuron = ()
)

@unpack TripodParam = neurons_quaresima2023

network = let
    # Number of neurons in the network
    # Define neurons and synapses in the network
	# proximal_distal = [(150um, 400um), (150um, 400um)], defines the dendrite dimensions later used in create_dendrite
    @unpack TripodParam =  neurons_quaresima2023
    E = SNN.BallAndStick((150um, 400um);
        N = TripodParam.N,
        soma_syn = TripodParam.soma_syn,
        dend_syn = TripodParam.dend_syn,
        NMDA = TripodParam.NMDA,
        param = TripodParam.param
    )

    # Define interneurons I1 and I2
    I1 = SNN.IF(; N = NI1, param = PVDuarte)

    I2 = SNN.IF(; N = NI2, param = SSTDuarte)
    # Define synaptic interactions between neurons and interneurons

    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; ConnectivityParams.IfE...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; ConnectivityParams.IsE...,)

    I1_to_E = SNN.CompartmentSynapse(
        I1,
        E,
        :s,
        :hi;
        param = plasticity_quaresima2023.iSTDP_rate,
        ConnectivityParams.EIf...,
    )

    I2_to_E = SNN.CompartmentSynapse( # what learning rate should I use?
        I2,
        E,
        :d,
        :hi;
        param = plasticity_quaresima2023.iSTDP_potential,
        ConnectivityParams.EdIs...,
    )

    # Define recurrent connections in the network
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; ConnectivityParams.IfIf...)
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; ConnectivityParams.IsIs...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; ConnectivityParams.IfIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; ConnectivityParams.IsIf...)

    E_to_E = SNN.CompartmentSynapse(
        E,
        E,
        :d,
        :he;
        param = SNNUtils.quaresima2023.vstdp,
        ConnectivityParams.EdE...,
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
    :noise_d1  => SNN.PoissonStimulus(network.pop.E,  :he_d, param=1kHz, cells=:ALL, μ=2.f0,),
    :noise_i1  => SNN.PoissonStimulus(network.pop.I1, :ge,   param=2.8kHz, cells=:ALL, μ=1.f0),
    :noise_i2  => SNN.PoissonStimulus(network.pop.I2, :ge,   param=2.8kHz, cells=:ALL, μ=4.f0),
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
SNN.vecplot!(p,model.pop.E, :v_d, r=3.5s:4.5s, neurons=cells, dt=0.125, pop_average=false)
myintervals = sign_intervals(:A, seq)
vline!(myintervals, c=:red)
##
model.syn.I2_to_E.W


pointer(model.pop.I1.name)
pointer(model.pop.E.name)
