# Four E/I assemblies (400 AdEx excitatory and 100 IF inhibitory neurons each) with inhibitory plasticity on I -> E,
# lateral E -> I connections between all assemblies and Poisson noise. After a first run, assembly 1 receives a
# 1 s pulse of 8.5 kHz Poisson input at 8 s and assembly 2 at 12 s. Plots the raster, the I -> E weights, the
# assembly rates and their correlation matrix.
# (The original used iSTDPTime, which no longer exists; it is replaced by iSTDPRate with a 4 Hz target.)
# Runtime: about 2 min.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie
using Statistics

# Define each of the network recurrent assemblies
function define_network(N, name, istdp)
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -55mV, b = 0pA, a = 0nS),
                 spike = SNN.PostSpike(At = 1mV), name = "Exc_$name")
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "Inh_$name")
    E_to_I = SNN.SpikingSynapse(E, I, :he; conn = (p = 0.2, μ = 1.0), name = "E_to_I_$name")
    E_to_E = SNN.SpikingSynapse(E, E, :he; conn = (p = 0.2, μ = 0.5), name = "E_to_E_$name")
    I_to_I = SNN.SpikingSynapse(I, I, :hi; conn = (p = 0.2, μ = 1.0), name = "I_to_I_$name")
    I_to_E = SNN.SpikingSynapse(I, E, :hi; conn = (p = 0.2, μ = 25), LTPParam = istdp, name = "I_to_E_$name")
    norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 10ms), [E_to_E])
    noise = SNN.PoissonStimulus(E, :he; param = SNN.PoissonFixed(rate = 4.5kHz), neurons = :ALL)
    SNN.monitor!([E, I], [:fire])
    SNN.monitor!(I_to_E, [:W], sr = 10Hz)
    return SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, norm, noise, silent = true)
end

n_assemblies = 4
istdp = SNN.iSTDPRate(τy = 20ms, η = 0.5, r = 4Hz)
subnets = NamedTuple(Symbol("sub_$n") => define_network(400, n, istdp) for n = 1:n_assemblies)
syns = NamedTuple(
    Symbol("$(i)E_to_$(j)I_lateral") => SNN.SpikingSynapse(
        subnets[i].pop.E, subnets[j].pop.I, :he; conn = (p = 0.2, μ = 2.0))
    for i = 1:n_assemblies for j = 1:n_assemblies if i != j
)

trigger = (
    first = SNN.PoissonStimulus(subnets[1].pop.E, :he;
        param = SNN.PoissonInterval(rate = 8.5kHz, intervals = [[8s, 9s]]), neurons = :ALL, name = "First stim"),
    second = SNN.PoissonStimulus(subnets[2].pop.E, :he;
        param = SNN.PoissonInterval(rate = 8.5kHz, intervals = [[12s, 13s]]), neurons = :ALL, name = "Second stim"),
)
network = SNN.compose(; subnets..., syns..., trigger, silent = true)

SNN.train!(model = network, duration = 15000ms, dt = 0.125f0)

# Plot the raster plot of the network
SNN.raster(network.pop, [0s, 15s], every = 5)

##
i_to_e = [network.syn.sub_1_I_to_E, network.syn.sub_2_I_to_E, network.syn.sub_3_I_to_E, network.syn.sub_4_I_to_E]
w_i = map(syn -> vec(mean(SNN.getvariable(syn, :W), dims = 1)), i_to_e)   # mean weight at each sample
_, r_t = SNN.record(i_to_e[1], :W, range = true)

fig = Figure(size = (800, 1000))
ax1 = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Synaptic weight", title = "I to E synapses")
for (i, w) in enumerate(w_i)
    lines!(ax1, r_t ./ 1000, w, label = "pop $i", linewidth = 3)
end
axislegend(ax1, position = :lt)

exc = [network.pop.sub_1_E, network.pop.sub_2_E, network.pop.sub_3_E, network.pop.sub_4_E]
interval = 0s:20ms:15s
rates = map(exc) do p
    fr, _ = SNN.firing_rate(p; interval, τ = 50ms, pop_average = true)
    vec(fr)
end
ax2 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Firing rate (Hz)", title = "Excitatory populations")
for (i, r) in enumerate(rates)
    lines!(ax2, collect(interval) ./ 1000, r, label = "Assembly $i", linewidth = 2)
end
axislegend(ax2, position = :lt)

after = findall(t -> t >= 2s, interval)
cor_mat = [cor(rates[i][after], rates[j][after]) for i in eachindex(rates), j in eachindex(rates)]
ax3 = Axis(fig[3, 1], xlabel = "Assembly", ylabel = "Assembly", title = "Correlation matrix", aspect = 1)
hm = heatmap!(ax3, cor_mat, colormap = :balance, colorrange = (-1, 1))
Colorbar(fig[3, 2], hm)
fig
