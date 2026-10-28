# Balanced network with the neuron parameters of Litwin-Kumar & Doiron 2014 (Nat. Commun. 5:5319):
# 4000 excitatory and 1000 inhibitory AdEx neurons, single-exponential synapses (τe = 6 ms, τi = 2 ms), sparse
# random connectivity (p = 0.2) and Poisson drive onto both populations. This is the network without plasticity
# (the vSTDP version is in LitwinKumar2014_AssociativeMemories/). Plots a raster and the synaptic
# conductances of one excitatory neuron.
# Runtime: about 40 s.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie, Statistics, Random

## LKD parameters
τm = 20ms
C = 300pF
R = τm / C          # R = 1 nS / gl, so gl = C / τm = 15 nS

LKD_AdEx_exc = SNN.AdExParameter(τm = τm, R = R, Vt = -52mV, Vr = -60mV, El = -70mV)
LKD_AdEx_inh = SNN.AdExParameter(τm = τm, R = R, Vt = -52mV, Vr = -60mV, El = -62mV)
synapse = SNN.SingleExpSynapse(τe = 6ms, τi = 2ms)

# inputs: every neuron receives a Poisson train of N ν = 1000 x 4.5 Hz (E) or 1000 x 2.5 Hz (I) with weight μ_in.
# The original script left μ_in undefined; 0.25 gives rates of about 5 Hz in both populations.
N_in = 1000
νe = N_in * 4.5Hz
νi = N_in * 2.5Hz
μ_in = 0.25

μEE = 2.76 / C * pF
μEI = 1.27 / C * pF
μIE = 48.7 / C * pF
μII = 16.2 / C * pF

E = SNN.AdEx(; N = 4000, param = LKD_AdEx_exc, synapse, name = "E")
I = SNN.AdEx(; N = 1000, param = LKD_AdEx_inh, synapse, name = "I")

EE = SNN.SpikingSynapse(E, E, :ge; conn = (μ = μEE, p = 0.2))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = μEI, p = 0.2))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (μ = μIE, p = 0.2))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = μII, p = 0.2))

Input_E = SNN.PoissonStimulus(E, :ge; param = SNN.PoissonFixed(rate = νe, μ = μ_in), neurons = :ALL)
Input_I = SNN.PoissonStimulus(I, :ge; param = SNN.PoissonFixed(rate = νi, μ = μ_in), neurons = :ALL)

##
model = SNN.compose(; E, I, Input_E, Input_I, EE, EI, IE, II, silent = true)
SNN.monitor!(model.pop, [:fire])
SNN.sim!(model, 15second)
SNN.raster(model.pop, [4.3s, 5s])

SNN.monitor!([E, I], [:ge, :gi], variables = :synvars)
SNN.monitor!([E, I], [:v])
SNN.sim!(model, 1second)
fig = Figure(size = (700, 350))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Conductance")
SNN.vecplot!(ax, E, :ge, variables = :synvars, neurons = 123, label = "ge")
SNN.vecplot!(ax, E, :gi, variables = :synvars, neurons = 123, label = "gi")
axislegend(ax)
fig
