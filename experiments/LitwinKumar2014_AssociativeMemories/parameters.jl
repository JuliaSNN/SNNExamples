# 
LKD_network = (
    Npop = (E=4000, I=1000),
    exc = AdExParameter(
                            El = -70mV, 
                            Vt = -52.0mV, 
                            τm = 300pF /15.0nS, 
                            R = 1/(15.0nS),
                            Vr = -60.0f0mV,
                            τabs = 1ms,       
                            τri=0.5,
                            τdi=2.0,
                            τre=1.0,
                            τde=6.0,
                            E_i = -75mV,
                            E_e = 0mV,
                            At = 10mV
                            ),
    inh = IFParameter(
            El = -62.0mV,
            Vr = -57.47mV,   #(mV)
            Vt = -52.0mV,
            τm = 20ms,
            a = 0.0,
            b = 0.0,
            τw = 144,
            τri=0.5,
            τdi=2.0,
            τre=1.0,
            τde=6.0,
        ),
    plasticity = (
        iSTDP_rate = SNN.iSTDPRate(η = 0.2, τy = 10ms, r=5Hz, Wmax = 200.0pF, Wmin = 2.78pF),
        vSTDP = SNN.vSTDPParameter(
            A_LTD = 4.0f-4,  #ltd strength
            A_LTP = 14.0f-4, #ltp strength
            Wmin = 0.01,  #minimum ee strength
            Wmax = 81.4,   #maximum ee strength
        )
    ),
    connections = (
        E_to_E = (p = 0.2, μ = 2.76),
        E_to_I = (p = 0.2, μ = 1.27),
        I_to_E = (p = 0.2, μ = 48.7),
        I_to_I = (p = 0.2, μ = 16.2),
        Th_to_E = (p=0.2, μ=2.0pA),
        ),
    noise = (rE=4.5kHz,rI=2.25kHz, μE=2.75pA, μI=2.75pA),
)