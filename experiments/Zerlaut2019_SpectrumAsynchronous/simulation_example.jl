# Balanced network of Zerlaut et al. (2019) with 8000 excitatory and 2000 inhibitory neurons under Poisson
# afferents of 4 Hz and 10 Hz: raster and population rates of the last 7 s of a 10 s simulation, for both rates.
# Runtime: about 60 s.
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units
include("model.jl")

config_big = SNN.@update Zerlaut2019_network begin
    Npop = (E = 8000, I = 2000)
end

fig = Figure(size = (1200, 600))
for (col, input_rate) in enumerate([4, 10])
    config = SNN.@update config_big begin
        afferents.rate = input_rate * Hz
    end
    model = soma_network(config)
    SNN.sim!(; model, duration = 10_000ms)

    # Raster of the network
    ax_r = Axis(fig[2, col], xlabel = "Time (s)", ylabel = "Neuron")
    SNN.raster!(ax_r, model.pop, 3s:10s, every = 40)

    # Firing rate of the network with a fixed afferent rate
    frE, r = SNN.firing_rate(model.pop.E, interval = 3s:10ms:10s, pop_average = true)
    frI, _ = SNN.firing_rate(model.pop.I, interval = 3s:10ms:10s, pop_average = true)
    ax_f = Axis(fig[1, col], ylabel = "Firing rate (Hz)", title = "Afferent rate: $input_rate Hz")
    lines!(ax_f, r ./ 1000, vec(frE), label = "E", linewidth = 2)
    lines!(ax_f, r ./ 1000, vec(frI), label = "I", linewidth = 2)
    axislegend(ax_f)
end
fig
