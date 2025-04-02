using Distributions

## Synapses Tripod neuron
bursty_dendritic_network2 = let
    DuarteGluSoma =  Glutamatergic(
            Receptor(E_rev = 0.0mV, τr = 0.26ms, τd = 2.0ms, g0 = 0.73nS), # AMPA
            Receptor(), # NMDA
        )
    MilesGabaSoma =  GABAergic(
            Receptor(E_rev = -70.0mV, τr = 0.1ms, τd = 15.0ms, g0 = 0.38nS), # GABAa
            Receptor() # GABAb
        )
    EyalGluDend = Glutamatergic(
            Receptor(E_rev = 0.0mV, τr = 0.26ms, τd = 2.0ms, g0 = 0.73nS), # AMPA
            Receptor(E_rev = 0.0mV, τr = 8ms, τd = 35.0ms, g0 = 1.31, nmda = 1.0f0),
        )
    MilesGabaDend =  GABAergic(
            Receptor(E_rev = -70.0mV, τr = 4.8ms, τd = 29.0ms, g0 = 0.27nS), # GABAa
            Receptor(E_rev = -90.0mV, τr = 30ms, τd = 400.0ms, g0 = 0.006nS), # GABAb
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
            τm = 281pF / 40nS, # * ms,  # membrane time constant
            Er = -70.6mV,  # resting potential
            Vr = -55mV, # reset potential # Vr = -70.6mV,
            Vt = -50mV,  # threshold potential # Vt = -50.4mV
            ΔT = 2mV,  # slope factor
            τw = 144ms,  # adaptation time constant
            a = 4nS,  # subthreshold adaptation conductance
            b = 80.5pA,  # spike-triggered adaptation current
            AP_membrane = 20.0f0mV,  # action potential membrane potential
            BAP = 1.0f0mV,  # burst afterpotential
            up = 1ms,  # refractory period
            τabs = 2ms,  # absolute refractory period
        ),
        soma_syn=  Synapse(DuarteGluSoma, MilesGabaSoma),  # connect EyalGluDend to MilesGabaDend
        dend_syn = Synapse(EyalGluDend, MilesGabaDend) # defines glutamaterbic and gabaergic receptors in the dendrites
    )
    PV = SNN.IFParameterGsyn(
        τm = 104.52pF / 9.75nS, #) * ms
        El = -64.33mV,
        Vt = -38.97mV,
        Vr = -57.47mV,
        E_i = -75mV,
        E_e = 0mV,
        τabs = 0.5ms,
        τre = 0.18ms,
        τde = 0.70ms,
        τri = 0.19ms,
        τdi = 2.50ms,
        gsyn_e = 1.04nS,
        gsyn_i = 0.84nS,
    )

    SST = SNN.IFParameterGsyn(
        τm = 102.86pF / 4.61nS, #) * ms,
        El = -61mV,
        Vt = -34.4mV,
        Vr = -47.11mV,
        E_i = -75mV,
        E_e = 0mV,
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
        iSTDP_rate = SNN.iSTDPParameterRate(η = 0.2, τy = 20ms, r=10Hz, Wmax = 243.0pF, Wmin = 2.78pF),
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.2, v0 = -70mV, τy = 5ms, Wmax = 243.0pF, Wmin = 2.78pF),
        # iSTDP_antihebbian = SNN.STDPAntiHebbianAsymmetric(τpre=5ms, τpost=5ms, A_post = 10e-1pA / mV, A_pre =  10e-1pA / (mV * mV), Wmax=15.0pF),
        vstdp = SNN.vSTDPParameter(
            A_LTD = 4.0f-4,  #ltd strength # CHANGED from 4.0f-5
            A_LTP = 1.4f-3, #ltp strength # CHANGED from 1.4f-4
            θ_LTD = -40.0mV,  #ltd voltage threshold # set higher
            θ_LTP = -20.0mV,  #ltp voltage threshold
            τu = 15.0ms,  #timescale for u variable
            τv = 45.0ms,  #timescale for v variable
            τx = 20.0ms,  #timescale for x variable
            Wmin = 2.78pF,  #minimum ee strength
            Wmax = 41.4pF,   #maximum ee strength
        )
    )
    connectivity = (
        E_to_Ed = (p = 0.2,  μ = 10.78, dist = Normal, σ = 1),
        E_to_If = (p = 0.2,  μ = log(15.27),  dist = LogNormal, σ = 0.),
        E_to_Is = (p = 0.2,  μ = log(15.27),  dist = LogNormal, σ = 0.),

        If_to_E = (p = 0.2,  μ = log(15.8), dist = LogNormal, σ = 0.),
        If_to_Is = (p = 0.2, μ = log(0.83),  dist = LogNormal, σ = 0.),
        If_to_If = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),

        Is_to_Ed = (p = 0.2, μ = log(15.8), dist = LogNormal, σ = 0.),
        Is_to_If = (p = 0.2, μ = log(1.47), dist = LogNormal, σ = 0.),
        Is_to_Is = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),

        EIig = (p = 0.2, μ = log(10.0), dist = LogNormal, σ = 0.), # ADDED
        EIic = (p = 0.2, μ = log(6.0), dist = LogNormal, σ = 0.), # ADDED
    )

    noise_params = let
        exc_soma = (param=4.5kHz,  μ=2.78f0pF,  N=500, neurons=:ALL, name="noise_exc_soma")
        exc_dend = (param=0.0kHz,  μ=0.f0,  N=500, neurons=:ALL, name="noise_exc_dend")
        inh1 = (param=2.25kHz,  μ=2.78f0pF,  N=500, neurons=:ALL,     name="noise_inh1")
        inh2 = (param=2.25kHz,  μ=2.78f0pF, N=500, neurons=:ALL,     name="noise_inh2")
        inhib = (param=2.25kHz,  μ=2.78f0pF, N=500, neurons=:ALL,     name="noise_inhib")
        (exc_soma=exc_soma, exc_dend=exc_dend, inh1=inh1, inh2=inh2, inhib=inhib)
    end

    inh_ratio = (
                    ni1 = 0.35 * 1/4,
                    ni2 = 0.65 * 1/4,
        )

    (exc=exc, pv=PV, sst=SST, plasticity, connectivity, noise_params, inh_ratio)
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
    E_to_E = SNN.SpikingSynapse(E, E, :he, :d ; connectivity.E_to_Ed..., param= plasticity.vstdp)
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.E_to_If...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.E_to_Is...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.If_to_E...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.If_to_If...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.Is_to_If...)
    I2_to_I2 = SNN.SpikingSynapse(I2, I2, :gi; connectivity.Is_to_Is...)
    I2_to_E = SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.Is_to_Ed...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.If_to_Is...)
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
    E_to_E1 = SNN.SpikingSynapse(E, E, :he, :d1 ; connectivity.E_to_Ed..., param= plasticity.vstdp)
    E_to_E2 = SNN.SpikingSynapse(E, E, :he, :d2 ; connectivity.E_to_Ed..., param= plasticity.vstdp)

    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.E_to_If...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.E_to_Is...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.If_to_E...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.If_to_If...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.Is_to_If...)
    I2_to_I2 = SNN.SpikingSynapse(I2, I2, :gi; connectivity.Is_to_Is...)
    I2_to_E1 = SNN.SpikingSynapse(I2, E, :hi, :d1; param = plasticity.iSTDP_potential, connectivity.Is_to_Ed...)
    I2_to_E2 = SNN.SpikingSynapse(I2, E, :hi, :d2; param = plasticity.iSTDP_potential, connectivity.Is_to_Ed...)

    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.If_to_Is...)
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
