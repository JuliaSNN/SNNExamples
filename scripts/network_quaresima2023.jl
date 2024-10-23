using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

## Define the network parameters


PVDuarte = SNN.IFParameter(
    τm = 104.52pF / 9.75nS,
    El = -64.33mV,
    Vt = -38.97mV,
    Vr = -57.47mV,
    τabs = 0.5ms,
    # gsyn_e = 1.04,
    # gsyn_i = 0.84,
    # τe = 0.70ms,
    # τi = 2.50ms
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
)

## ! Make them adaptive with these parameters (and verify the others)
# PVDuarte = LIF(
#     Er = -64.33,
#     u_r = -57.47,   #(mV)
#     θ = -38.97,   #(mV)
#     C = 104.52,    #(pF)
#     gl = 9.75,       #(nS)
#     a = 0.0,       #(nS) 'sub-threshold' adaptation conductance
#     b = 10.0,       #(pA) 'sra' current increment
#     τw = 144,        #(s) adaptation time constant (~Ca-activated K current inactivation)
#     idle = 0.52,       #(ms)
# )

# SSTDuarte = LIF(
#     Er = -61,
#     u_r = -47.11,
#     θ = -34.4,
#     C = 102.87,     #(pF)
#     gl = 4.61,       #(nS)
#     a = 4.0,        #(nS) 'sub-threshold' adaptation conductance
#     b = 80.5,       #(pA) 'sra' current increment
#     τw = 144,        #(s) adaptation time constant (~Ca-activated K current inactivation)
#     idle = 1.34,       #(ms)
# )

ConnectivityParams = (
    EdE = (p = 0.2,  μ = 10.8, dist = Normal, σ = 1),
    IfE = (p = 0.2,  μ = log(5.7), dist = LogNormal, σ = 0.1),
    IsE = (p = 0.2,  μ = log(5.7), dist = LogNormal, σ = 0.1),
    EIf = (p = 0.2,  μ = log(15.8), dist = LogNormal, σ = 0),
    IsIf = (p = 0.2, μ = log(1.4),  dist = LogNormal, σ = 0.25),
    IfIf = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.14),
    EdIs = (p = 0.2, μ = log(15.8), dist = LogNormal, σ = 0),
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


network = let
    # Number of neurons in the network
    NE = 1000
    NI2 = 175
    NI1 = 325
    # Define neurons and synapses in the network
	# proximal_distal = [(150um, 400um), (150um, 400um)], defines the dendrite dimensions later used in create_dendrite
    E = SNN.Tripod(proximal_distal...;
        N = NE,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma), # defines glutamaterbic and gabaergic receptors in the soma
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        NMDA = SNN.EyalNMDA,
        param = SNN.AdExSoma(Vr = -55mV, Vt = -50mV),
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
        :inh;
        param = plasticity_quaresima2023.iSTDP_rate,
        ConnectivityParams.EIf...,
    )

    I2_to_E_d1 = SNN.CompartmentSynapse( # what learning rate should I use?
        I2,
        E,
        :d1,
        :inh;
        param = plasticity_quaresima2023.iSTDP_potential,
        ConnectivityParams.EdIs...,
    )

    I2_to_E_d2 = SNN.CompartmentSynapse( # what learning rate should I use?
        I2,
        E,
        :d2,
        :inh;
        param = plasticity_quaresima2023.iSTDP_potential,
        ConnectivityParams.EdIs...,
    )

    # Define recurrent connections in the network
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; ConnectivityParams.IfIf...)
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; ConnectivityParams.IsIs...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; ConnectivityParams.IfIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; ConnectivityParams.IsIf...)

    E_to_E1_d1 = SNN.CompartmentSynapse(
        E,
        E,
        :d1,
        :exc;
        param = SNNUtils.quaresima2023.vstdp,
        ConnectivityParams.EdE...,
    )

    E_to_E2_d2 = SNN.CompartmentSynapse(
        E,
        E,
        :d2,
        :exc;
        param = SNNUtils.quaresima2023.vstdp,
        ConnectivityParams.EdE...,
    )

    # Define normalization
    recurrent_norm =
    [
        SNN.SynapseNormalization(NE, [E_to_E1_d1], param = SNN.MultiplicativeNorm(τ = 20ms)),
        SNN.SynapseNormalization(NE, [E_to_E2_d2], param = SNN.MultiplicativeNorm(τ = 20ms))
    ]

    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E_d1 I2_to_E_d2 I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E1_d1 E_to_E2_d2 norm1=recurrent_norm[1] norm2=recurrent_norm[2])
    
	# Return the network as a tuple
    (pop = pop, syn = syn)
end



## Stimulus
# Background noise
init_noise(x)= x < 1s ? 1000Hz - x : 0.0
stimuli = Dict(
    :noise_s   => SNN.PoissonStimulus(network.pop.E,  :g_s,  param=PSParam(rate= (x,y)->4.5kHz), cells=:ALL, μ=2.7f0,),
    :noise_d1  => SNN.PoissonStimulus(network.pop.E,  :g_d1, param=PSParam(rate= (x,y)->2.5kHz), cells=:ALL, μ=2.f0,),
    :noise_d2  => SNN.PoissonStimulus(network.pop.E,  :g_d2, param=PSParam(rate= (x,y)->2.5kHz), cells=:ALL, μ=2.f0,),
    :noise_i1  => SNN.PoissonStimulus(network.pop.I1, :ge,   param=PSParam(rate= (x,y)->2.5kHz), cells=:ALL, μ=1.f0),
    :noise_i2  => SNN.PoissonStimulus(network.pop.I2, :ge,   param=PSParam(rate= (x,y)->2.5kHz), cells=:ALL, μ=3.8f0),
)

baseline = merge_models(stimuli, network)


## Sequence input
dictionary = Dict(:AB=>[:A, :B], :BA=>[:B, :A])
duration = Dict(:A=>40, :B=>60, :_=>200)
config = (seq_length=100, silence=1, dictionary=dictionary, ph_duration=duration, init_silence=1s)
seq = generate_sequence(config)


function step_input(x, param::PSParam) 
    intervals::Vector{Vector{Float32}} = param.variables[:intervals]
    # intervals = param.variables[:intervals]
    if time_in_interval(x, intervals)
        return 6000Hz
    else
        return 0.0
    end
end

stim_d1 = Dict{Symbol,Any}()
stim_d2 = Dict{Symbol,Any}()
for w in seq.symbols.words
    param = PSParam(rate=step_input, variables=Dict(:intervals=>sign_intervals(seq.string2id[w], seq)))
    push!(stim_d1,w  => SNN.PoissonStimulus(network.pop.E, :h_d1, μ=5.f0, receptors=[1,2], param=param))
    push!(stim_d2,w  => SNN.PoissonStimulus(network.pop.E, :h_d2, μ=5.f0, receptors=[1,2], param=param))
end
for p in seq.symbols.phonemes
    param = PSParam(rate=step_input, variables=Dict(:intervals=>sign_intervals(seq.string2id[p], seq)))
    push!(stim_d1,p  => SNN.PoissonStimulus(network.pop.E, :h_d1, μ=5.f0, receptors=[1,2], param=param))
    push!(stim_d2,p  => SNN.PoissonStimulus(network.pop.E, :h_d2, μ=5.f0, receptors=[1,2], param=param))
end

stim_d1 = dict2ntuple(stim_d1)
stim_d2 = dict2ntuple(stim_d2)
model = merge_models(baseline, d1=stim_d1, d2=stim_d2)


stim_d1.A.param.rate(10f0,stim_d1.A.param)

##
SNN.monitor(model.pop.E, [:fire, :v_s, :v_d1, :v_d2, :h_s, :h_d1, :h_d2, :g_d1, :g_d2])
SNN.monitor(model.pop.I1, [:fire, :v, :ge, :gi])
SNN.monitor(model.pop.I2, [:fire, :v, :ge, :gi])
SNN.monitor([network.pop...], [:fire])
duration = sequence_end(seq)
SNN.train!(model=model, duration= 5s, pbar=true, dt=0.125)
SNN.raster([network.pop...], (0000,5000))

## Target activation with stimuli
p = plot()
cells = collect(Set(stim_d1.AB.cells))
SNN.vecplot!(p,model.pop.E, :v_d1, r=2.5s:4.5s, neurons=cells, dt=0.125, pop_average=true)
myintervals = sign_intervals(seq.string2id[:AB], seq)
vline!(myintervals, c=:red)
##


mean(length.(SNN.spiketimes(model.pop.E)))/5
