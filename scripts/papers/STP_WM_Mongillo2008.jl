using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

MongilloParam = (
        Exc = IFCurrentDeltaParameter(
        R =1,
        τm = 15ms,
        Vt = 20mV,
        Vr = 16mV,
        El = 0mV,
        τabs = 2ms,
    ),
        Inh = IFCurrentDeltaParameter(
        R =1,
        τm = 10ms,
        Vt = 20mV,
        Vr = 13mV,
        El = 0mV,
        τabs = 2ms,
    )
)

pop = (
    E = IFCurrent(N=4000, param=MongilloParam.Exc),
    I = IFCurrent(N=1000, param=MongilloParam.Inh)
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
    E = SNN.CurrentStimulus(pop.E, I_dist=Normal(21.10, 1.0), I_base=0.0, α=1.0),
    I = SNN.CurrentStimulus(pop.I, I_dist=Normal(21.0, 1.0), I_base=0.0, α=1.0)
)

model = SNN.merge_models(pop, syn, stim)
SNN.monitor([model.pop...], [:fire])
SNN.monitor(model.syn.EE, [:u])

time_keeper = SNN.Time()
# @profview 
SNN.train!(model=model,duration=1s, dt=0.125, pbar=true, time=time_keeper)

SNN.vecplot(pop.I,:v, r=990:0.01:1s, dt=0.125, neurons=1:10)
SNN.vecplot(pop.E,:v, r=1:0.01:0.4s, dt=0.125, neurons=1:1)
SNN.vecplot(syn.EE, :u, r=1:0.01:1s, dt=0.125, neurons=1:1)
SNN.vecplot(syn.EE, :x, r=1:0.01:1s, dt=0.125, neurons=1:1)

SNN.raster(pop, [0.95, 1.0] .* 1s)

mean(length.(SNN.spiketimes(pop.E)))