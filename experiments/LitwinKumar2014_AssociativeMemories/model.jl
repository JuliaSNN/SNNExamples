# Network of Litwin-Kumar & Doiron (2014): 4000 AdEx excitatory and 1000 IF inhibitory neurons with vSTDP on E -> E,
# inhibitory STDP on I -> E, multiplicative synaptic normalisation and Poisson background input.
# `soma_network(config)` takes a named tuple with the fields of `LKD_network` (parameters.jl).
function soma_network(config)
    (; noise, connections, Npop, synapse) = config
    E = SNN.AdEx(N = Npop.E, param = config.exc, spike = config.exc_spike, synapse = synapse, name = "E")
    I = SNN.IF(N = Npop.I, param = config.inh, synapse = synapse, name = "I")

    noiseE = SNN.PoissonStimulus(E, :he; neurons = :ALL, param = SNN.PoissonFixed(rate = noise.rE, μ = noise.μE), name = "noiseE")
    noiseI = SNN.PoissonStimulus(I, :he; neurons = :ALL, param = SNN.PoissonFixed(rate = noise.rI, μ = noise.μI), name = "noiseI")

    (; iSTDP_rate, vSTDP) = config.plasticity
    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :he; conn = connections.E_to_E, name = "E_to_E", LTPParam = vSTDP),
        E_to_I = SNN.SpikingSynapse(E, I, :he; conn = connections.E_to_I, name = "E_to_I"),
        I_to_E = SNN.SpikingSynapse(I, E, :hi; conn = connections.I_to_E, name = "I_to_E", LTPParam = iSTDP_rate),
        I_to_I = SNN.SpikingSynapse(I, I, :hi; conn = connections.I_to_I, name = "I_to_I"),
    )
    scaling = SNN.MetaPlasticity(SNN.MultiplicativeNorm(τ = 20ms), [synapses.E_to_E])
    model = SNN.compose(; E, I, noiseI, noiseE, synapses..., scaling, silent = true, name = "Balanced network")
    SNN.monitor!(model.pop, [:fire])
    return model
end
