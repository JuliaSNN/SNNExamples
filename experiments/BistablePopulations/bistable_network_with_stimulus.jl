# Two E/I assemblies that inhibit each other through lateral E -> I connections and are driven by Poisson noise,
# plus a slowly alternating stimulus: during each second one assembly (the one whose id matches) receives an
# extra Poisson input that decays with a time constant of 200 ms. 5 s of warm-up, 15 s recorded. Plots the raster,
# the rate of each assembly with the correlation matrix, and the autocorrelation of the first assembly rate.
# Runtime: about 1.5 min.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie
using Statistics, StatsBase

# Define each of the network recurrent assemblies
function define_network(N, name)
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -55mV, b = 0pA, a = 0nS),
                 spike = SNN.PostSpike(At = 0mV), name = "E_$name")
    I = SNN.IF(; N = N ÷ 2, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "I_$name")
    E_to_I = SNN.SpikingSynapse(E, I, :he; conn = (p = 0.2, μ = 2.0))
    E_to_E = SNN.SpikingSynapse(E, E, :he; conn = (p = 0.2, μ = 3.0))
    I_to_I = SNN.SpikingSynapse(I, I, :hi; conn = (p = 0.2, μ = 1.0))
    I_to_E = SNN.SpikingSynapse(I, E, :hi; conn = (p = 0.2, μ = 1.0), LTPParam = SNN.iSTDPRate(r = 4Hz))
    norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 10ms), [E_to_E])
    noise = SNN.PoissonStimulus(E, :he; param = SNN.PoissonFixed(rate = 2.5kHz), neurons = :ALL)
    return SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, norm, noise, silent = true)
end

n_assemblies = 2
## Instantiate the network assemblies and local inhibitory populations
subnets = NamedTuple(Symbol("sub_$n") => define_network(400, n) for n = 1:n_assemblies)
# Lateral connections: the excitatory neurons of an assembly excite the inhibitory neurons of the other
syns = NamedTuple(
    Symbol("lateral_$(i)E_to_$(j)I") => SNN.SpikingSynapse(
        subnets[i].pop.E, subnets[j].pop.I, :he; conn = (p = 0.2, μ = 2.25))
    for i = 1:n_assemblies for j = 1:n_assemblies if i != j
)

# Alternating stimulus: in the second t ÷ 1000 the assembly with id (t ÷ 1000) % n + 1 receives 5 kHz decaying with 200 ms
function input(t, variables)
    id = variables[:id]
    n = variables[:n_assemblies]
    return (t ÷ 1000) % n + 1 == id ? 5kHz * exp(-(t % 1000) / 200) : 0.0f0
end
stimuli = NamedTuple(
    Symbol("stim_$n") => SNN.PoissonStimulus(
        subnets[n].pop.E, :ge; neurons = :ALL,
        param = SNN.PoissonVariable(rate = input, variables = Dict{Symbol,Any}(:id => n, :n_assemblies => n_assemblies)),
    ) for n = 1:n_assemblies
)
network = SNN.compose(; subnets..., syns..., stimuli..., silent = true)

## Warm-up, then record
SNN.train!(model = network, duration = 5000ms, dt = 0.125f0)
SNN.monitor!(network.pop, [:fire])
SNN.train!(model = network, duration = 15000ms, dt = 0.125f0)
SNN.raster(network.pop, [4s, 15s])

## Rate of each assembly, correlation matrix and autocorrelation
exc = [network.pop.sub_1_E, network.pop.sub_2_E]
interval = 5s:5ms:SNN.get_time(network)
rates = map(exc) do p
    fr, _ = SNN.firing_rate(p; interval, τ = 10ms, pop_average = true)
    vec(fr)
end
cor_mat = [cor(rates[i], rates[j]) for i in eachindex(rates), j in eachindex(rates)]

fig = Figure(size = (700, 800))
ax1 = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Firing rate (Hz)")
for i in eachindex(rates)
    lines!(ax1, collect(interval) ./ 1000, rates[i], label = "Assembly $i")
end
axislegend(ax1, position = :lt)
ax2 = Axis(fig[2, 1], xlabel = "Assembly", ylabel = "Assembly", title = "Correlation matrix", aspect = 1)
hm = heatmap!(ax2, cor_mat, colormap = :balance, colorrange = (-1, 1))
Colorbar(fig[2, 2], hm)
ax3 = Axis(fig[3, 1], xlabel = "Lag (ms)", ylabel = "Autocorrelation")
nlag = 800
lines!(ax3, (0:nlag) .* 5, autocor(rates[1], 0:nlag))
fig
