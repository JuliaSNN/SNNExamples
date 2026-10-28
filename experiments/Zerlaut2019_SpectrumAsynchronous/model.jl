# Balanced E/I network with the parameters of Zerlaut et al. (2019, J. Comput. Neurosci. 46:1): conductance-based
# IF neurons with single-exponential synapses, 4000 excitatory and 1000 inhibitory neurons, sparse connectivity
# (p = 0.05) and Poisson afferents (100 afferent neurons, p = 0.1). `soma_network(config)` builds the model and
# records the spikes of the populations and of the afferents. Included by the simulation scripts.
using SpikingNeuralNetworks
SNN.@load_units

Zerlaut2019_network = (
    Npop = (E = 4000, I = 1000),
    exc = SNN.IFParameter(C = 200pF, gl = 10nS, El = -70mV, Vt = -50.0mV, Vr = -70.0mV),
    inh = SNN.IFParameter(C = 200pF, gl = 10nS, El = -70mV, Vt = -53.0mV, Vr = -70.0mV),
    spike = SNN.PostSpike(τabs = 2ms),
    synapse = SNN.SingleExpSynapse(τi = 5ms, τe = 5ms, E_i = -80mV, E_e = 0mV),
    connections = (
        E_to_E = (p = 0.05, μ = 2nS),
        E_to_I = (p = 0.05, μ = 2nS),
        I_to_E = (p = 0.05, μ = 10nS),
        I_to_I = (p = 0.05, μ = 10nS),
    ),
    afferents = (N = 100, rate = 20Hz, conn = (p = 0.1f0, μ = 4.0)),
)

function soma_network(config)
    (; afferents, connections, Npop, synapse, spike) = config
    E = SNN.IF(; N = Npop.E, param = config.exc, synapse, spike, name = "E")
    I = SNN.IF(; N = Npop.I, param = config.inh, synapse, spike, name = "I")

    AfferentParam = SNN.PoissonLayer(rate = afferents.rate, N = afferents.N)
    afferentE = SNN.Stimulus(AfferentParam, E, :ge, conn = afferents.conn, name = "noiseE")
    afferentI = SNN.Stimulus(AfferentParam, I, :ge, conn = afferents.conn, name = "noiseI")

    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :ge, conn = connections.E_to_E, name = "E_to_E"),
        E_to_I = SNN.SpikingSynapse(E, I, :ge, conn = connections.E_to_I, name = "E_to_I"),
        I_to_E = SNN.SpikingSynapse(I, E, :gi, conn = connections.I_to_E, name = "I_to_E"),
        I_to_I = SNN.SpikingSynapse(I, I, :gi, conn = connections.I_to_I, name = "I_to_I"),
    )
    model = SNN.compose(; E, I, afferentE, afferentI, synapses..., silent = true, name = "Balanced network")
    SNN.monitor!(model.pop, [:fire])
    SNN.monitor!(model.stim, [:fire])
    return model
end
