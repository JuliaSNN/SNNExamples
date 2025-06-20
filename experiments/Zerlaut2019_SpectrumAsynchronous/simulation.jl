using DrWatson
using Distributions
using YAML
using ThreadTools

using SpikingNeuralNetworks
using SNNUtils
using SNNPlots
SNN.@load_units
include(projectdir("loaders", "filesystem.jl"))
begin
    include("model.jl")
    plots_path = plotsdir("Zerlaut2019") |> mkpath
    data_path = datadir("Zerlaut2019") |> mkpath  
end
#


νa =  exp.(range(log(1), log(40), 20))
f_rate = map(νa) do x
    frs = tmap(1:5) do _
        config = @update Zerlaut2019_network begin
            afferents.rate = x*Hz
        end 
        model = soma_network(config)
        sim!(;model, duration=10_000ms,  pbar=false)
        fr, _ = firing_rate(model.pop.E, interval=3s:10s, pop_average=true, time_average=true)
    end
        f =     mean(frs)
    @info "rate: $x Hz = $f"
    frs
end
##

ff_rate = [filter(x -> x < 80, mean.(fr)) for fr in f_rate]
SNNPlots.scatter(νa, mean.(ff_rate), ribbon=std.(ff_rate), scale=:log10, xlims=(1,20), ylims=(0.00001,80))#, xscale=:log, yscale=:log)
SNNPlots.plot!(νa, mean.(ff_rate), ribbon=std.(ff_rate), scale=:log10, xlims=(1, 20), ylims=(0.00001,80), lw=5, xticks=([1,5,10, 20], [1,5,10,20]))#, xscale=:log, yscale=:log)
plot!(xlabel="Afferent rate (Hz)", ylabel="Firing rate (Hz)",  legend=false, size=(400,400))









# ##
# # raster(model.pop, 0s:1s, every=10)
# # vecplot(model.pop.E, :v, interval=0s:0.5s, neurons=1:1)
# # plot(fr)


# # raster(model.stim, 0s:1s)
# ##


# model = soma_network(Zerlaut2019_network)
# sim!(;model, duration=3000ms,  pbar=true)
# fr, r= firing_rate(model.pop.E, interval=2s:3s, pop_average=true)

# plot(fr)
# ##
# raster(model.pop, 0s:3s, every=1)

# fr, r= firing_rate(model.pop.E, interval=2s:3s, pop_average=true)
# vecplot(model.pop.E, :v, interval=0s:1s, neurons=1:1)
# plot(fr)
# ##

# length(model.stim.afferentE.W) /100

# model.stim.afferentE.I


