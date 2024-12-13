using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using StatsBase
using Distributions

MongilloParam = (
        Exc = IFCurrentDeltaParameter(
        R =1,
        τm = 15ms,
        Vt = 20mV,
        Vr = 16mV,
        τabs = 2ms,
    ),
        Inh = IFCurrentDeltaParameter(
        R =1,
        τm = 10ms,
        Vt = 20mV,
        Vr = 13mV,
        τabs = 2ms,
    )
)

pop = (
    E = IFCurrent(N=8000, param=MongilloParam.Exc),
    I = IFCurrent(N=2000, param=MongilloParam.Inh)
)

syn = (
    EE = SpikingSynapse(pop.E, pop.E, :ge, p=0.2, μ=0.10, 
        param=SNN.STPParameter(), 
        delay_dist=Uniform(1ms,5ms)),
    EI = SpikingSynapse(pop.E, pop.I, :ge, p=0.2, μ=0.135, delay_dist=Uniform(1ms,5ms)),
    IE = SpikingSynapse(pop.I, pop.E, :gi, p=0.2, μ=0.5, delay_dist=Uniform(1ms,5ms)),
    II = SpikingSynapse(pop.I, pop.I, :gi, p=0.2, μ=0.20, delay_dist=Uniform(1ms,5ms)),
)

stim = (
    E = SNN.CurrentStimulus(pop.E, I_dist=Normal(30.10, 1.0), I_base=0.0, α=1.0),
    I = SNN.CurrentStimulus(pop.I, I_dist=Normal(21.0, 1.0), I_base=0.0, α=1.0)
)

model = SNN.merge_models(pop, syn, stim)
# n_assemblies = 1
# n_cells = 100
# w_assembly = 0.48
# assemblies= map(1:n_assemblies) do x
#     # cells = StatsBase.sample(1:pop.E.N, n_cells, replace=false)
#     cells = 1:80 |> collect
#     update_weights!(syn.EE, cells, cells, w_assembly)
#     (cells=cells, name=Symbol("assembly$x"))
# end

# attack_rate = 20Hz
# decay_rate = 10ms
# peak_rate = 15kHz
# stim_parameters = Dict(:decay=>decay_rate, :peak=>peak_rate, :start=>attack_rate)
# stim_assembly = Dict( assembly.name=>begin
#                             variables = merge(stim_parameters, Dict(:intervals=>[[3s, 3s+0.2s], [3s+0.8s, 3s+1s]]))
#                             param = PSParam(rate=attack_decay, 
#                                         variables=variables)
#                             SNN.PoissonStimulus(pop.E, :ge, μ=20pF, cells=assembly.cells, param=param, name=string(assembly.name))
#                         end 
#                     for assembly in assemblies)

##
# model = SNN.merge_models(model, stim_assembly)
SNN.monitor([model.pop...], [:fire, :v])
SNN.monitor(model.syn.EE, [:u, :x])

SNN.train!(model=model,duration=5s, dt=0.125, pbar=true)
SNN.raster(pop, [2.4, 3.5] * 1s)
SNN.vecplot( pop.E, :v, r=1s:0.01:5s, neurons=1, pop_average=true)

##
interval = 2s:20ms:5s
fr, _ = SNN.firing_rate(pop, interval=interval)
plot(interval, mean(fr[1], dims=1)')
plot!(interval, mean(fr[2], dims=1)')
##
SNN.train!(model=model,duration=5s, dt=0.125, pbar=true, time=time_keeper)
# SNN.vecplot(pop.I,:v, r=990:0.01:1s, dt=0.125, neurons=1:10)
# SNN.vecplot(pop.E,:v, r=1:0.01:0.4s, dt=0.125, neurons=1:1)
p = SNN.vecplot(syn.EE, :u, r=1s:0.01:5s, dt=0.125, neurons=assemblies[1].cells,pop_average=true)
SNN.vecplot!(p, syn.EE, :x, r=1s:0.01:5s, dt=0.125, neurons=assemblies[1].cells,pop_average=true)
plot(p, ylims=:auto)
##


assemblies[1].cells
SNN.raster(pop.E, [3, 3.1] * 1s, populations=[a.cells for a in assemblies], names=["assembly1", "assembly2"])
interval = 2s:20ms:5s
fr, interval = SNN.firing_rate(pop.E, interval=interval)
plot(interval, mean(fr[assemblies[1].cells, interval], dims=1)')
plot!(interval, mean(fr[:, interval], dims=1)')
# cor(mean(fr[assembly1,interval], dims=1)',mean(fr[assembly2,interval], dims=1)')
