# 
Zerlaut2019_network = (
    Npop = (E=4000, I=1000),

    exc = IFParameterSingleExponential(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -50.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV,
                ),

    inh = IFParameterSingleExponential(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -53.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV,
                ),

    connections = (
        E_to_E = (p = 0.05, μ = 2nS),
        E_to_I = (p = 0.05, μ = 2nS),
        I_to_E = (p = 0.05, μ = 10nS),
        I_to_I = (p = 0.05, μ = 10nS),
        ),
    
    afferents = (
        N = 100,
        ϵ = 0.1f0,
        rate = 20Hz,
        μ = 4.0,
        ), 
    # noise = (rE=4.5kHz,rI=2.25kHz, μE=2.75pA, μI=2.75pA),
)

function soma_network(config)
    @unpack afferents, connections, Npop = config
    E = IF(N=Npop.E, param=config.exc, name="E")
    I = IF(N=Npop.I, param=config.inh, name="I")

    AfferentParam = PoissonStimulusLayer(afferents.N; rate=afferents.rate, ϵ=afferents.ϵ)
    afferentE = PoissonLayer(E, :ge, μ=afferents.μ, param=AfferentParam, name="noiseE")
    afferentI = PoissonLayer(I, :ge, μ=afferents.μ, param=AfferentParam, name="noiseI")

    # afferentE = PoissonStimulus(E, :ge, neurons=:ALL, μ=afferents.μ, param=3kHz, name="noiseE")
    # afferentI = PoissonStimulus(I, :ge, neurons=:ALL, μ=afferents.μ, param=4kHz, name="noiseI")

    synapses = (
        E_to_E = SpikingSynapse(E, E, :ge, p=connections.E_to_E.p, μ=connections.E_to_E.μ, name="E_to_E"),
        E_to_I = SpikingSynapse(E, I, :ge, p=connections.E_to_I.p, μ=connections.E_to_I.μ, name="E_to_I"),
        I_to_E = SpikingSynapse(I, E, :gi, p=connections.I_to_E.p, μ=connections.I_to_E.μ, name="I_to_E"),
        I_to_I = SpikingSynapse(I, I, :gi, p=connections.I_to_I.p, μ=connections.I_to_I.μ, name="I_to_I"),
    )
    model = merge_models(;E,I, afferentE, afferentI, synapses..., silent=true, name="Balanced network") 
    # scaling = SNN.SynapseNormalization(E, [synapses.E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    monitor!(model.pop, [:fire])
    monitor!(model.stim, [:fire])
    monitor!(model.pop, [:v], sr=200Hz)
    return merge_models(;model...)
end
