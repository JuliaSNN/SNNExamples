using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using StatsBase
using Distributions
using LaTeXStrings

##

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
        El = 0mV,
        Vr = 13mV,
        τabs = 2ms,
        # τe = 1ms,
        # τi = 1ms
    )
)

pop = (
    E = IFCurrent(N=8, param=MongilloParam.Exc),
)

μee = 1 #* 8000/pop.E.N
μee_assembly = 1# 8000/pop.E.N
μei =  1 # 8000/pop.E.N
μie =  1
μii =  1
input_exc = 20.8
input_inh = 19.8

syn = (
    EE = SpikingSynapse(pop.E, pop.E, :ge, p=1., σ=0, μ=μee, delay_dist=Uniform(1ms,5ms), param=SNN.STPParameter()),
)

stim = (
    E = SNN.CurrentStimulus(pop.E, I_dist=Normal(input_exc, 1.0), α=1.0),
)

model = SNN.merge_models(pop, syn, stim)

SNN.monitor([model.pop...], [:fire, :v], sr=50Hz)
SNN.monitor(model.syn.EE, [:u, :x, :g], sr=1000Hz)
simtime = SNN.train!(model=model,duration=1s, dt=0.125, pbar=true)
##
raster(model.pop.E, [0,1s])

plot(sum(getvariable(syn.EE, :g), dims=1)[1,:])

SNN.integrate!(pop.E, pop.E.param, 0.125f0)
SNN.forward!(syn.EE, syn.EE.param)
syn.EE.g .=1