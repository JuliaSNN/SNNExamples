# Paradoxical effect of inhibition-stabilised networks (Tsodyks et al. 1997, J. Neurosci. 17:4382),
# first version of the experiment (2000 AdEx excitatory and 500 IF inhibitory neurons with single-exponential
# synapses, Poisson drive onto E). A current into the I neurons is swept from 0 to 1 nA and the steady-state
# rates of both populations are plotted. See tutorials/networks/ParadoxicalEffect.jl for the version with
# conductance-based parameters of the paper.
# Runtime: about 40 s.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie, Statistics, Random

function initialize()
    synapse = SNN.SingleExpSynapse()
    E = SNN.AdEx(; N = 2000, param = SNN.AdExParameter(El = -65mV), synapse, name = "E")
    I = SNN.IF(; N = 500, param = SNN.IFParameter(), synapse, name = "I")
    EE = SNN.SpikingSynapse(E, E, :ge; conn = (μ = 1.0, p = 0.2))
    EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 5.0, p = 0.2))
    IE = SNN.SpikingSynapse(I, E, :gi; conn = (μ = 10.0, p = 0.2))
    II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10.0, p = 0.2))
    S = SNN.PoissonStimulus(E, :ge; param = SNN.PoissonFixed(rate = 800Hz, μ = 1.7f0), neurons = :ALL)
    return SNN.compose(; E, I, EE, EI, IE, II, S, silent = true)
end

## raster and rates without extra current
model = initialize()
SNN.monitor!(model.pop, [:fire, :v, :ge])
SNN.sim!(model, 3second)
SNN.raster(model.pop, [1s, 3s])
@show mean(SNN.firing_rate(model.pop.I, interval = 1s:10ms:3s, time_average = true)[1])
@show mean(SNN.firing_rate(model.pop.E, interval = 1s:10ms:3s, time_average = true)[1])

## sweep of the current into the inhibitory population
Irange = 0:0.10nA:1nA
rI = zeros(length(Irange))
rE = zeros(length(Irange))
for (n, x) in enumerate(Irange)
    Random.seed!(10)
    model = initialize()
    SNN.sim!(model, 1second)
    SNN.monitor!(model.pop, [:fire])
    model.pop.I.I .= x
    SNN.sim!(model, 1second)
    rI[n] = mean(SNN.firing_rate(model.pop.I, interval = 1s:10ms:2s, time_average = true)[1])
    rE[n] = mean(SNN.firing_rate(model.pop.E, interval = 1s:10ms:2s, time_average = true)[1])
    @info "E: $(rE[n]) Hz I: $(rI[n]) Hz"
end

fig = Figure(size = (550, 380))
ax1 = Axis(fig[1, 1], xlabel = "Input to I neurons (nA)", ylabel = "Inh. firing rate (Hz)",
           yticklabelcolor = :blue, ylabelcolor = :blue)
ax2 = Axis(fig[1, 1], ylabel = "Exc. firing rate (Hz)", yaxisposition = :right,
           yticklabelcolor = :red, ylabelcolor = :red)
hidespines!(ax2); hidexdecorations!(ax2)
lines!(ax1, collect(Irange) ./ nA, rI, color = :blue, linewidth = 4)
lines!(ax2, collect(Irange) ./ nA, rE, color = :red, linewidth = 4)
fig
