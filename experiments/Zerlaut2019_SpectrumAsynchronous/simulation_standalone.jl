# Transfer function of the balanced network of Zerlaut et al. (2019), 4000 excitatory and 1000 inhibitory neurons:
# mean excitatory rate over the last 7 s of a 10 s simulation, against the afferent rate (20 values between
# 1 and 20 Hz, 5 repetitions each, run on the available threads). Rates above 80 Hz are discarded as in the
# original script. The model is included from model.jl (the original script carried its own copy).
# Runtime: about 2 min.
using ThreadTools
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units
include("model.jl")

νa = exp.(range(log(1), log(20), 20))
f_rate = map(νa) do x
    frs = tmap(1:5) do _
        config = SNN.@update Zerlaut2019_network begin
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

ff_rate = [filter(x -> x < 80, fr) for fr in f_rate]
fig = Figure(size = (450, 450))
ax = Axis(fig[1, 1], xlabel = "Afferent rate (Hz)", ylabel = "Firing rate (Hz)", xscale = log10, yscale = log10,
          xticks = ([1, 5, 10, 20], ["1", "5", "10", "20"]))
m = max.(mean.(ff_rate), 1e-6)
s = std.(ff_rate)
band!(ax, νa, max.(m .- s, 1e-6), m .+ s, alpha = 0.3)
scatterlines!(ax, νa, m, linewidth = 3)
xlims!(ax, 0.9, 22)
fig
