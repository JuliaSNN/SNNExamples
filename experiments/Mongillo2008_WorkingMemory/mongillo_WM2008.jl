# Network of Mongillo, Barak & Tsodyks (2008, Science 319:1543): 8000 excitatory and 2000 inhibitory
# integrate-and-fire neurons with delta (current-based) synapses, random connectivity (p = 0.2), delays
# U(1, 5) ms, short-term plasticity (U = 0.2, tau_D = 200 ms, tau_F = 1500 ms) on the E -> E synapses, and
# a noisy external current. `Mongillo2008(; n_assemblies, n_neurons)` returns the model and the strengthened
# E assemblies that hold an item in synaptic working memory. Included by STP_WM_Mongillo2008.jl.
using Distributions, StatsBase
using SpikingNeuralNetworks
SNN.@load_units

const MongilloParam = (
    Exc = SNN.IFParameter(R = 1, τm = 15ms, Vt = 20mV, Vr = 16mV, El = 0mV),
    Inh = SNN.IFParameter(R = 1, τm = 10ms, Vt = 20mV, Vr = 13mV, El = 0mV),
    spike = SNN.PostSpike(τabs = 2ms),
    synapse = SNN.DeltaSynapse(),
)

function Mongillo2008(; n_assemblies = 1, n_neurons = 800)
    (; Exc, Inh, spike, synapse) = MongilloParam
    pop = (
        E = SNN.IF(; N = 8000, param = Exc, spike, synapse, name = "E"),
        I = SNN.IF(; N = 2000, param = Inh, spike, synapse, name = "I"),
    )

    connections = (;
        EE = (p = 0.2, σ = 0, μ = 0.10 * 8000 / pop.E.N),
        EI = (p = 0.2, σ = 0, μ = 0.135 * 8000 / pop.E.N),
        IE = (p = 0.2, σ = 0, μ = 0.25 * 2000 / pop.I.N),
        II = (p = 0.2, σ = 0, μ = 0.20 * 2000 / pop.I.N),
    )

    input_exc = 19.8
    input_inh = 19.8

    syn = (
        EE = SNN.SpikingSynapse(
            pop.E,
            pop.E,
            :ge,
            conn = connections.EE,
            STPParam = SNN.MarkramSTPParameter(τD = 200ms, τF = 1500ms, U = 0.2),
            delay_dist = Uniform(1ms, 5ms),
        ),
        EI = SNN.SpikingSynapse(pop.E, pop.I, :ge, conn = connections.EI, delay_dist = Uniform(1ms, 5ms)),
        IE = SNN.SpikingSynapse(pop.I, pop.E, :gi, conn = connections.IE, delay_dist = Uniform(1ms, 5ms)),
        II = SNN.SpikingSynapse(pop.I, pop.I, :gi, conn = connections.II, delay_dist = Uniform(1ms, 5ms)),
    )

    stim = (
        E = SNN.CurrentStimulus(pop.E, param = SNN.CurrentNoise(pop.E; I_dist = Normal(input_exc, 1.0), α = 0.0)),
        I = SNN.CurrentStimulus(pop.I, param = SNN.CurrentNoise(pop.I; I_dist = Normal(input_inh, 1.0), α = 0.0)),
    )

    μee_assembly = 0.48 * 8000 / pop.E.N
    model = SNN.compose(pop, syn, stim)
    assemblies = map(1:n_assemblies) do x
        neurons = StatsBase.sample(1:pop.E.N, n_neurons, replace = false)
        SNN.update_weights!(syn.EE, neurons, neurons, μee_assembly)
        (
            neurons = neurons,
            name = Symbol("assembly$x"),
            indices = SNN.indices(syn.EE, neurons, neurons),
            id = x,
        )
    end
    return model, assemblies
end
