# Simulation speed of three single-neuron models (AdEx, Tripod, BallAndStick) driven by Poisson layers:
# `@btime sim!(model, 10s)` for each. The numbers depend on the machine; none is plotted.
# Runtime: about 25 s.
using SNNModels
using BenchmarkTools
using Random
@load_units

## AdEx neuron with 1000 excitatory and 1000 inhibitory Poisson inputs
E = Population(AdExParameter(), N = 100, synapse = DoubleExpSynapse(), spike = PostSpike())

poisson_exc = PoissonLayer(10.2Hz, N = 1000)   # mean firing rate and number of neurons in the layer
poisson_inh = PoissonLayer(3Hz, N = 1000)

exc_conn = (p = 1.0f0, μ = 1.0f0)   # connection probability, synaptic strength (nS)
inh_conn = (p = 1.0f0, μ = 4.0f0)
stim_exc = Stimulus(poisson_exc, E, :glu, conn = exc_conn, name = "noiseE")
stim_inh = Stimulus(poisson_inh, E, :gaba, conn = inh_conn, name = "noiseI")

model = compose(; E, stim_exc, stim_inh, silent = true)
println("AdEx (100 neurons):")
@btime sim!($model, 10s) samples = 3 evals = 1
#   240.436 ms (799501 allocations: 15.86 MiB) on DellTower (older SNNModels, for comparison only)

## Tripod neuron, Poisson input on both dendrites
Random.seed!(1234)
dend_neuron = (;
    param = SNNModels.TripodParameter(),
    spike = SNNModels.PostSpike(At = 1, τA = 30ms, τabs = 10, up = 10),
    adex = SNNModels.AdExParameter(Vr = -40mV, b = 80pA, a = 0),
    dend_syn = SNNModels.TripodDendSynapse,
    soma_syn = SNNModels.TripodSomaSynapse,
)
E = SNNModels.Population(; dend_neuron..., N = 1)

poisson_exc = PoissonLayer(10.2Hz, N = 200)
poisson_inh = PoissonLayer(3Hz, N = 1000)
exc_conn = (p = 1.0f0, μ = 8.5)
inh_conn = (p = 1.0f0, μ = 5.0)
stim_exc1 = Stimulus(poisson_exc, E, :glu, :d1, conn = exc_conn, name = "noiseE1")
stim_inh1 = Stimulus(poisson_inh, E, :gaba, :d1, conn = inh_conn, name = "noiseI1")
stim_exc2 = Stimulus(poisson_exc, E, :glu, :d2, conn = exc_conn, name = "noiseE2")
stim_inh2 = Stimulus(poisson_inh, E, :gaba, :d2, conn = inh_conn, name = "noiseI2")

model = compose(; E, stim_exc1, stim_inh1, stim_exc2, stim_inh2, silent = true)
println("Tripod:")
@btime sim!($model, 10s) samples = 3 evals = 1
#   238.624 ms (1119507 allocations: 20.75 MiB) on DellTower (older SNNModels, for comparison only)

## Ball-and-stick neuron, Poisson input on the dendrite
Random.seed!(1234)
dend_neuron = (;
    param = SNNModels.BallAndStickParameter(),
    spike = SNNModels.PostSpike(At = 1, τA = 30ms, τabs = 10, up = 10),
    adex = SNNModels.AdExParameter(Vr = -40mV, b = 80pA, a = 0),
    dend_syn = SNNModels.TripodDendSynapse,
    soma_syn = SNNModels.TripodSomaSynapse,
)
E = SNNModels.Population(; dend_neuron..., N = 1)

stim_exc = Stimulus(poisson_exc, E, :glu, :d, conn = exc_conn, name = "noiseE")
stim_inh = Stimulus(poisson_inh, E, :gaba, :d, conn = inh_conn, name = "noiseI")

model = compose(; E, stim_exc, stim_inh, silent = true)
println("BallAndStick:")
@btime sim!($model, 10s) samples = 3 evals = 1
#   735.852 ms (799501 allocations: 15.86 MiB) on DellTower (older SNNModels, for comparison only)
