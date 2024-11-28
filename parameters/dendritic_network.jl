## Synapses Tripod neuron
bursty_dendritic_network = (
    exc = (
        dends = [(150um, 400um), (150um, 400um)],  # dendritic lengths
        NMDA = NMDAVoltageDependency(
            b = 3.36,  # NMDA voltage dependency parameter
            k = -0.077,  # NMDA voltage dependency parameter
            mg = 1.0f0,  # NMDA voltage dependency parameter
        ),
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
        dend_syn = let
            MilesGabaSoma =
                GABAergic(Receptor(E_rev = -75.0, τr = 0.5, τd = 6.0, g0 = 0.265), Receptor())  # GABAergic synapse from MilesGabaSoma to soma
            DuarteGluSoma = Glutamatergic(
                Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73), 
                ReceptorVoltage(E_rev = 0.0, nmda = 0.0f0),)  # Glutamatergic synapse from DuarteGluSoma to soma
            Synapse(DuarteGluSoma, MilesGabaSoma)  # connect DuarteGluSoma to MilesGabaSoma
            end,
        soma_syn= let
            EyalGluDend = Glutamatergic(
                Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73),
                ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0))  # Glutamatergic synapse from EyalGluDend to dendrite
            MilesGabaDend = GABAergic(
                Receptor(E_rev = -75.0, τr = 4.8, τd = 29.0, g0 = 0.126),
                Receptor(E_rev = -90.0, τr = 30, τd = 100.0, g0 = 0.006),)  # GABAergic synapse from MilesGabaDend to dendrite
            Synapse(EyalGluDend, MilesGabaDend)  # connect EyalGluDend to MilesGabaDend
        end,
    ),
    pv = SNN.IFParameterGsyn(
        τm = 104.52pF / 9.75nS,  # membrane time constant
        El = -64.33mV,  # leak reversal potential
        Vt = -38.97mV,  # threshold potential
        Vr = -57.47mV,  # reset potential
        τabs = 0.42ms,  # absolute refractory period
        τre = 0.18ms,  # excitatory recovery time constant
        τde = 0.70ms,  # excitatory decay time constant
        τri = 0.2ms,  # inhibitory recovery time constant
        τdi = 2.50ms,  # inhibitory decay time constant
    ),
    sst = SNN.IFParameterGsyn(
        τm = 102.86pF / 4.61nS,  # membrane time constant
        El = -61mV,  # leak reversal potential
        Vt = -34.4mV,  # threshold potential
        Vr = -47.11mV,  # reset potential
        τabs = 1.34ms,  # absolute refractory period
        τre = 0.18ms,  # excitatory rise synaptic time constant
        τde = 1.80ms,  # excitatory decay synapstic time constant
        τri = 0.19ms,  # inhibitory recovery time constant
        τdi = 5.00ms,  # inhibitory decay time constant
        gsyn_e = 0.8,  # excitatory synaptic conductance
        gsyn_i = 0.7,  # inhibitory synaptic conductance
        b = 80.5,       #(pA) 'sra' current increment
        τw = 144,        #(s) adaptation time constant (~Ca-activated K current inactivation)
    ),

    plasticity = (
        iSTDP_rate = SNN.iSTDPParameterRate(η = 0.2, τy = 5ms, r=10Hz, Wmax = 273.4pF, Wmin = 0.1pF),  # iSTDP rate-based plasticity parameters
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.2, v0 = -70mV, τy = 20ms, Wmax = 273.4pF, Wmin = 0.1pF),  # iSTDP potential-based plasticity parameters
        vstdp = SNN.vSTDPParameter(
                A_LTD = 4.0f-5,  # long-term depression strength
                A_LTP = 14.0f-5, # long-term potentiation strength
                θ_LTD = -40.0,  # long-term depression voltage threshold
                θ_LTP = -20.0,  # long-term potentiation voltage threshold
                τu = 15.0,  # timescale for u variable
                τv = 45.0,  # timescale for v variable
                τx = 20.0,  # timescale for x variable
                Wmin = 1.78,  # minimum excitatory-excitatory strength
                Wmax = 41.4,   # maximum excitatory-excitatory strength
            )
    ),
    connectivity = (
        EdE = (p = 0.2,  μ = 10., dist = Normal, σ = 1), 
        IfE = (p = 0.2,  μ = log(2.0),  dist = LogNormal, σ = 0.), 
        IsE = (p = 0.2,  μ = log(2.5),  dist = LogNormal, σ = 0.), 

        EIf = (p = 0.2,  μ = log(10.8), dist = LogNormal, σ = 0), 
        IsIf = (p = 0.2, μ = log(2.4),  dist = LogNormal, σ = 0.25),  
        IfIf = (p = 0.2, μ = log(15.2), dist = LogNormal, σ = 0.14),  

        EdIs = (p = 0.2, μ = log(10.0), dist = LogNormal, σ = 0),  
        IfIs = (p = 0.2, μ = log(5.83), dist = LogNormal, σ = 0.1),  
        IsIs = (p = 0.2, μ = log(5.83), dist = LogNormal, σ = 0.1),  
    )
)

bursty_dendritic_network.exc.param