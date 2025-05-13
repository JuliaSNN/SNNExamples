using DrWatson
using Distributions
using YAML

using SpikingNeuralNetworks
using SNNPlots
using SNNUtils
SNN.@load_units
include(projectdir("loaders", "filesystem.jl"))
begin
    include("model.jl")
    plots_path = plotsdir("Zerlaut2019") |> mkpath
    data_path = datadir("Zerlaut2019") |> mkpath  
end
#


exp.(log.(1:20))
f_rate = map(0:2.5:50) do x
    config = @update Zerlaut2019_network begin
        afferents.rate = x*Hz
    end 
    model = soma_network(config)
    sim!(;model, duration=10_000ms,  pbar=true)
    fr, r= firing_rate(model.pop.E, interval=3s:10s, pop_average=true)
    f = mean(fr)
    @info "rate: $x Hz = $f"
    f
end

plot(f_rate)#, xscale=:log, yscale=:log)
##
# raster(model.pop, 0s:1s, every=10)
# vecplot(model.pop.E, :v, interval=0s:0.5s, neurons=1:1)
# plot(fr)


# raster(model.stim, 0s:1s)
##


model = soma_network(Zerlaut2019_network)
sim!(;model, duration=3000ms,  pbar=true)
fr, r= firing_rate(model.pop.E, interval=2s:3s, pop_average=true)

plot(fr)
##
raster(model.pop, 0s:3s, every=1)

fr, r= firing_rate(model.pop.E, interval=2s:3s, pop_average=true)
vecplot(model.pop.E, :v, interval=0s:1s, neurons=1:1)
plot(fr)
##

length(model.stim.afferentE.W) /100

model.stim.afferentE.I


