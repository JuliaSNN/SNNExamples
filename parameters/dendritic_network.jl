using Distributions

## Synapses Tripod neuron
bursty_dendritic_network = let 
    EyalGluDend = Glutamatergic(
                Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
                ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
            )
    DuarteGluSoma =  Glutamatergic(
            Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73), 
            ReceptorVoltage(E_rev = 0.0, nmda = 0.0f0),
        )
    MilesGabaDend =  GABAergic(
            Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27), 
            Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), # τd = 100.0
        )
    exc = (
        dends = [(150um, 400um), (150um, 400um)],  # dendritic lengths
        NMDA = NMDAVoltageDependency(
            b = 3.36,  # NMDA voltage dependency parameter
            k = -0.077,  # NMDA voltage dependency parameter
            mg = 1.0f0,  # NMDA voltage dependency parameter
        ),
            # After spike timescales and membrane
        param= AdExSoma(
            C = 281pF,  # membrane capacitance
            gl = 40nS,  # leak conductance
            R = nS / 40nS * SNN.GΩ,  # membrane resistance
            τm = 281pF / 40nS,  # membrane time constant
            Er = -70.6mV,  # reset potential
            Vr = -55.6mV,  # resting potential
            Vt = -50.4mV,  # threshold potential
            ΔT = 2mV,  # slope factor
            τw = 144ms,  # adaptation time constant
            a = 4nS,  # subthreshold adaptation conductance
            b = 10.5pA,  # spike-triggered adaptation current
            AP_membrane = 2.0f0mV,  # action potential membrane potential
            BAP = 1.0f0mV,  # burst afterpotential
            up = 1ms,  # refractory period
            τabs = 2ms,  # absolute refractory period
        ),
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        soma_syn=  Synapse(DuarteGluSoma, MilesGabaSoma)  # connect EyalGluDend to MilesGabaDend
    )
    PV = SNN.IFParameterGsyn(
        τm = 104.52pF / 9.75nS,
        El = -64.33mV,
        Vt = -38.97mV,
        Vr = -57.47mV,
        τabs = 0.5ms, 
        τre = 0.18ms,
        τde = 0.70ms,
        τri = 0.19ms,
        τdi = 2.50ms,
        gsyn_e = 1.04nS,
        gsyn_i = 0.84nS, 
    )

    SST = SNN.IFParameterGsyn(
        τm = 102.86pF / 4.61nS,
        El = -61mV,
        Vt = -34.4mV,
        Vr = -47.11mV,
        τabs = 1.3ms,
        τre = 0.18ms,
        τde = 1.80ms,
        τri = 0.19ms,
        τdi = 5.00ms,
        gsyn_e = 0.56nS, 
        gsyn_i = 0.59nS, 
        a = 4nS,
        b = 80.5pA,       #(pA) 'sra' current increment
        τw = 144ms,        #(s) adaptation time constant (~Ca-activated K current inactivation)
    )
    plasticity = (
        iSTDP_rate = SNN.iSTDPParameterRate(η = 0.2, τy = 10ms, r=10Hz, Wmax = 200.0pF, Wmin = 2.78pF),
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.2, v0 = -70mV, τy = 20ms, Wmax = 200.0pF, Wmin = 2.78pF),
        vstdp = SNN.vSTDPParameter(
                A_LTD = 4.0f-4,  #ltd strength
                A_LTP = 14.0f-4, #ltp strength
                θ_LTD = -40.0,  #ltd voltage threshold # set higher
                θ_LTP = -20.0,  #ltp voltage threshold
                τu = 15.0,  #timescale for u variable
                τv = 45.0,  #timescale for v variable
                τx = 20.0,  #timescale for x variable
                Wmin = 2.78,  #minimum ee strength
                Wmax = 81.4,   #maximum ee strength
            )
    )
    connectivity = (
        EdE = (p = 0.2,  μ = 10.78, dist = Normal, σ = 1),
        IfE = (p = 0.2,  μ = log(15.27),  dist = LogNormal, σ = 0.),
        IsE = (p = 0.2,  μ = log(15.27),  dist = LogNormal, σ = 0.),

        EIf = (p = 0.2,  μ = log(15.8), dist = LogNormal, σ = 0.),
        IsIf = (p = 0.2, μ = log(0.83),  dist = LogNormal, σ = 0.),
        IfIf = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),

        EdIs = (p = 0.2, μ = log(15.8), dist = LogNormal, σ = 0.),
        IfIs = (p = 0.2, μ = log(1.47), dist = LogNormal, σ = 0.),
        IsIs = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),
    )

    noise_params = let
        exc_soma = (param=4.0kHz,  μ=2.8f0,  neurons=:ALL, name="noise_exc_soma")
        exc_dend = (param=0.0kHz,  μ=0.f0,  neurons=:ALL, name="noise_exc_dend")
        inh1 = (param=2.5kHz,  μ=2.8f0,  neurons=:ALL,     name="noise_inh1")
        inh2 = (param=3.5kHz,  μ=2.8f0, neurons=:ALL,     name="noise_inh2")
        (exc_soma=exc_soma, exc_dend=exc_dend, inh1=inh1, inh2=inh2)
    end

    inh_ratio = (
                    ni1 = 0.35 *1/4,
                    ni2 = 0.65 *    1/4,    
        )

    (exc=exc, pv=PV, sst=SST, plasticity,connectivity, noise_params, inh_ratio)
end

function ballstick_network(;params, name, NE, STDP, kwargs...)
    @unpack exc, pv, sst, plasticity, connectivity=params
    @unpack  noise_params, inh_ratio = params
    exc = quaresima2022
    # pv = duarte2019.PV
    # sst = duarte2019.SST
    # @unpack connectivity, plasticity = quaresima2023
    @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    # Number of neurons in the network
    NI1 = round(Int,NE * inh_ratio.ni1)
    NI2 = round(Int,NE * inh_ratio.ni2)
    # Import models parameters
    # Define interneurons I1 and I2
    # @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    E = SNN.BallAndStickHet(; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="ExcBallStick")
    I1 = SNN.IF(; N = NI1, param = pv, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = sst, name="I2_sst")
    # Define synaptic interactions between neurons and interneurons
    E_to_E = SNN.SpikingSynapse(E, E, :he, :d ; connectivity.EdE..., param= plasticity.vstdp)
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.EIf...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...)
    I2_to_I2 = SNN.SpikingSynapse(I2, I2, :gi; connectivity.IsIs...)
    I2_to_E = SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.EdIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)
    # Define normalization
    norm = SNN.SynapseNormalization(NE, [E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    # background noise
    @unpack exc_soma, exc_dend, inh1, inh2= noise_params
    stimuli = Dict(
        :s   => SNN.PoissonStimulus(E,  :he_s; exc_soma... ),
        :d   => SNN.PoissonStimulus(E,  :he_d; exc_dend... ),
        :i1  => SNN.PoissonStimulus(I1, :ge;   inh1...  ),
        :i2  => SNN.PoissonStimulus(I2, :ge;   inh2...  )
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    # Return the network as a model
    model = merge_models(pop, syn, noise=stimuli, name=name)
    if !STDP
        syn.E_to_E.param.active[1] = false
    end
    @info "STDP enabled: $(has_plasticity(model.syn.E_to_E1))"
    return model
end

function tripod_network(;params, name, NE, STDP, kwargs...)
    @unpack exc, pv, sst, plasticity, connectivity=params
    @unpack  noise_params, inh_ratio = params
    exc = quaresima2022
    # pv = duarte2019.PV
    # sst = duarte2019.SST
    # @unpack connectivity, plasticity = quaresima2023
    @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    # Number of neurons in the network
    NI1 = round(Int,NE * inh_ratio.ni1)
    NI2 = round(Int,NE * inh_ratio.ni2)
    # Import models parameters
    # Define interneurons I1 and I2
    # @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    E = SNN.TripodHet(; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="ExcTripod")
    I1 = SNN.IF(; N = NI1, param = pv, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = sst, name="I2_sst")
    # Define synaptic interactions between neurons and interneurons
    E_to_E1 = SNN.SpikingSynapse(E, E, :he, :d1 ; connectivity.EdE..., param= plasticity.vstdp)
    E_to_E2 = SNN.SpikingSynapse(E, E, :he, :d2 ; connectivity.EdE..., param= plasticity.vstdp)

    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.EIf...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...)
    I2_to_I2 = SNN.SpikingSynapse(I2, I2, :gi; connectivity.IsIs...)
    I2_to_E1 = SNN.SpikingSynapse(I2, E, :hi, :d1; param = plasticity.iSTDP_potential, connectivity.EdIs...)
    I2_to_E2 = SNN.SpikingSynapse(I2, E, :hi, :d2; param = plasticity.iSTDP_potential, connectivity.EdIs...)

    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)
    # Define normalization
    norm1 = SNN.SynapseNormalization(NE, [E_to_E1], param = SNN.MultiplicativeNorm(τ = 20ms))
    norm2 = SNN.SynapseNormalization(NE, [E_to_E2], param = SNN.MultiplicativeNorm(τ = 20ms))
    # background noise
    @unpack exc_soma, exc_dend, inh1, inh2= noise_params
    stimuli = Dict(
        :s   => SNN.PoissonStimulus(E,  :he_s; exc_soma... ),
        :i1  => SNN.PoissonStimulus(I1, :ge;   inh1...  ),
        :i2  => SNN.PoissonStimulus(I2, :ge;   inh2...  )
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E1 I2_to_E2 I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E1 E_to_E2 norm1 norm2 )
    # Return the network as a model
    model = merge_models(pop, syn, noise=stimuli, name=name)
    if !STDP
        syn.E_to_E1.param.active[1] = false
        syn.E_to_E2.param.active[1] = false
    end
    @info "STDP enabled: $(has_plasticity(model.syn.E_to_E1))"
    return model
end
