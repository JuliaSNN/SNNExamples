# Transfer function of the balanced network of Zerlaut et al. (2019) with 8000 excitatory and 2000 inhibitory
# neurons: lower quintile (over 5 repetitions) of the excitatory rate over the last 7 s of a 10 s simulation, against
# the afferent rate (20 values between 1 and 20 Hz). Parallel over afferent rates with `tmap`.
# Runtime: about 7.5 min.
using ThreadTools
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units
include("model.jl")

config_big = SNN.@update Zerlaut2019_network begin
    Npop = (E = 8000, I = 2000)
end

νa = exp.(range(log(1), log(20), 20))
f_rate = tmap(νa) do x
    frs = map(1:5) do _
        config = SNN.@update config_big begin
            afferents.rate = x * Hz
        end
        model = soma_network(config)
        SNN.sim!(; model, duration = 10_000ms)
        fr, _ = SNN.firing_rate(model.pop.E, interval = 3s:10ms:10s, pop_average = true, time_average = true)
        fr
    end
    @info "rate: $x Hz = $(mean(frs))"
    frs
end

q = [quantile(fr, 0.2) for fr in f_rate]
fig = Figure(size = (450, 450))
ax = Axis(fig[1, 1], xlabel = "Afferent rate (Hz)", ylabel = "Firing rate (Hz)", xscale = log10, yscale = log10,
          xticks = ([1, 5, 10, 20], ["1", "5", "10", "20"]))
scatterlines!(ax, νa, max.(q, 1e-5), linewidth = 3)
xlims!(ax, 0.9, 22)
fig
