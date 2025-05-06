function soma_network(config)
    @unpack noise, connections, Npop = config
    E = AdEx(N=Npop.E, param=config.exc, name="E")
    I = IF(N=Npop.I, param=config.inh, name="I")

    noiseE = PoissonStimulus(E, :he, neurons=:ALL, μ=noise.μE, param=noise.rE, name="noiseE")
    noiseI = PoissonStimulus(I, :he, neurons=:ALL, μ=noise.μI, param=noise.rI, name="noiseI")

    @unpack iSTDP_rate, vSTDP = config.plasticity
    synapses = (
        E_to_E = SpikingSynapse(E, E, :he, p=connections.E_to_E.p, μ=connections.E_to_E.μ, name="E_to_E",LTPParam=NoSTDP),
        E_to_I = SpikingSynapse(E, I, :he, p=connections.E_to_I.p, μ=connections.E_to_I.μ, name="E_to_I"),
        I_to_E = SpikingSynapse(I, E, :hi, p=connections.I_to_E.p, μ=connections.I_to_E.μ, name="I_to_E",LTP=iSTDP_rate),
        I_to_I = SpikingSynapse(I, I, :hi, p=connections.I_to_I.p, μ=connections.I_to_I.μ, name="I_to_I"),
    )
    model = merge_models(;E,I, noiseI, noiseE, synapses..., silent=true, name="Balanced network") 
    scaling = SNN.SynapseNormalization(E, [synapses.E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    monitor!(model.pop, [:fire])
    return merge_models(;model..., scaling)
end