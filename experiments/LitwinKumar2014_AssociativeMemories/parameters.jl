# Parameters of the network of Litwin-Kumar & Doiron (2014, Nat. Commun. 5:5319): AdEx excitatory neurons with an
# adaptive threshold, IF inhibitory neurons, double-exponential conductance synapses (the default
# `DoubleExpSynapse`: τre = 1, τde = 6, τri = 0.5, τdi = 2 ms, E_i = -75 mV), voltage-based STDP on E -> E and
# inhibitory STDP on I -> E.
# Weights: the paper gives synaptic strengths in pF (J_EE = 2.76, J_EI = 1.27, J_IE = 48.7, J_II = 16.2). They are
# divided by the membrane capacitance C = 300 pF, so that they are conductances in nS in SNNModels 2 (the plasticity
# bounds likewise), and the external input weights (0.25 onto E, 0.18 onto I) were set by hand so that E and I fire at about 5 Hz.
# Neither choice has been checked against the original simulations.
C_LKD = 300pF
LKD_network = (
    Npop = (E = 4000, I = 1000),
    exc = SNN.AdExParameter(
        El = -70mV,
        Vt = -52.0mV,
        τm = 300pF / 15.0nS,
        R = 1 / (15.0nS),
        Vr = -60.0f0mV,
    ),
    exc_spike = SNN.PostSpike(τabs = 1ms, At = 10mV),
    inh = SNN.IFParameter(
        El = -62.0mV,
        Vr = -57.47mV,
        Vt = -52.0mV,
        τm = 20ms,
        R = 1 / (15.0nS),
    ),
    synapse = SNN.DoubleExpSynapse(τre = 1ms, τde = 6ms, τri = 0.5ms, τdi = 2ms, E_i = -75mV, E_e = 0mV),
    plasticity = (
        iSTDP_rate = SNN.iSTDPRate(η = 0.2 / C_LKD * pF, τy = 10ms, r = 5Hz, Wmax = 200.0pF / C_LKD, Wmin = 2.78pF / C_LKD),
        vSTDP = SNN.vSTDPParameter(
            A_LTD = 4.0f-4,  # LTD strength
            A_LTP = 14.0f-4, # LTP strength
            Wmin = 0.01 / C_LKD * pF, # minimum E -> E strength
            Wmax = 81.4 / C_LKD * pF, # maximum E -> E strength
        ),
    ),
    connections = (
        E_to_E = (p = 0.2, μ = 2.76 / C_LKD * pF),
        E_to_I = (p = 0.2, μ = 1.27 / C_LKD * pF),
        I_to_E = (p = 0.2, μ = 48.7 / C_LKD * pF),
        I_to_I = (p = 0.2, μ = 16.2 / C_LKD * pF),
        Th_to_E = (p = 0.2, μ = 2.0pA / C_LKD),
    ),
    noise = (rE = 4.5kHz, rI = 2.25kHz, μE = 0.25, μI = 0.18),
)
