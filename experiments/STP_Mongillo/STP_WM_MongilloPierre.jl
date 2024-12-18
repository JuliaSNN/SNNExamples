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
    )
)

pop = (
    E = IFCurrent(N=8000, param=MongilloParam.Exc),
    I = IFCurrent(N=2000, param=MongilloParam.Inh)
)

μee = 0.10 * 8000/pop.E.N
μee_assembly = 0.48 * 8000/pop.E.N  * 10
μei = 0.135 * 8000/pop.E.N
μie = 0.25 * 2000/pop.I.N
μii = 0.20 * 2000/pop.I.N
input_exc = 19.8
input_inh = 19.8

syn = (
    EE = SpikingSynapse(pop.E, pop.E, :ge, p=0.2, σ=0, μ=μee, param=SNN.STPParameter(), delay_dist=Uniform(1ms,5ms)),
    EI = SpikingSynapse(pop.E, pop.I, :ge, p=0.2, σ=0, μ=μei, delay_dist=Uniform(1ms,5ms)),
    IE = SpikingSynapse(pop.I, pop.E, :gi, p=0.2, σ=0, μ=μie, delay_dist=Uniform(1ms,5ms)),
    II = SpikingSynapse(pop.I, pop.I, :gi, p=0.2, σ=0, μ=μii, delay_dist=Uniform(1ms,5ms)),
)

stim = (
    E = SNN.CurrentStimulus(pop.E, I_dist=Normal(input_exc, 1.0), α=1.0),
    I = SNN.CurrentStimulus(pop.I, I_dist=Normal(input_inh, 1.0), α=1.0)
)

model = SNN.merge_models(pop, syn, stim)

n_assemblies = 1
n_cells = 200
assemblies= map(1:n_assemblies) do x
    cells = StatsBase.sample(1:pop.E.N, n_cells, replace=false)
    # cells = 1:100 |> collect
    update_weights!(syn.EE, cells, cells, μee_assembly)
    (cells=cells, name=Symbol("assembly$x"), indices= indices(syn.EE, cells, cells))
end



peak_rate = 3kHz
stim_parameters = Dict(:decay=>1ms, :peak=>peak_rate, :start=>peak_rate)
stim_assembly = Dict( assembly.name=>begin
                            variables = merge(stim_parameters, Dict(:intervals=>[[3s, 3s+2s]]))
                            param = PSParam(rate=attack_decay, 
                                        variables=variables)
                            SNN.PoissonStimulus(pop.E, :ge, μ=1pF, cells=assembly.cells, param=param, name=string(assembly.name))
                        end 
                    for assembly in assemblies)

model = SNN.merge_models(model, stim_assembly)
SNN.monitor([model.pop...], [:fire, :v], sr=50Hz)
SNN.monitor(model.syn.EE, [:u, :x], sr=50Hz)
w_rec = [assemblies[1].indices..., indices(syn.EE, 81:160, 81:160)...]
SNN.monitor(model.syn.EE, [(:ρ, w_rec ), (:W, w_rec )], sr=5Hz)

pop.E.records
simtime = SNN.train!(model=model,duration=5s, dt=0.125, pbar=true)
ρ, r_t= SNN.interpolated_record(syn.EE, :ρ)
w,r_t= SNN.interpolated_record(syn.EE, :W)
weff = ρ.*w ./μee_assembly
root = datadir("working_memory", "Mongillo2008") |> x-> (mkpath(dirname(x)); x)
path = SNN.save_model(path= root, model= model, name="8000_cells", info=MongilloParam, assemblies=assemblies, simtime=simtime)
interval = 0s:10ms:4.5s
stp_plot(model, interval, assemblies)
##

@unpack model, simtime, assemblies =  SNN.load_data(path)
simtime = SNN.train!(model=model,duration=2s, dt=0.125, pbar=true, time=simtime)
model.stim.E.I_base.=0.05pA
SNN.train!(model=model,duration=3s, dt=0.125, pbar=true, time=simtime)
interval = 2s:10ms:10s
stp_plot(model, interval, assemblies)
