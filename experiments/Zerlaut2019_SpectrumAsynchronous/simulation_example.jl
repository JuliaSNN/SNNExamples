using Pkg
Pkg.add("ThreadTools")
Pkg.add("Plots")
Pkg.add("UnPack")
Pkg.add("Statistics")
Pkg.add("Plots")
Pkg.add(url="https://github.com/JuliaSNN/SpikingNeuralNetworks.jl")

##
using ThreadTools
using Plots
using UnPack
using Statistics
using SpikingNeuralNetworks
SNN.@load_units

Zerlaut2019_network = (
    Npop = (E=4000, I=1000),

    exc = IFParameterSingleExponential(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -50.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV,
                ),

    inh = IFParameterSingleExponential(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -53.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV,
                ),

    connections = (
        E_to_E = (p = 0.05, μ = 2nS),
        E_to_I = (p = 0.05, μ = 2nS),
        I_to_E = (p = 0.05, μ = 10nS),
        I_to_I = (p = 0.05, μ = 10nS),
        ),
    
    afferents = (
        N = 100,
        p = 0.1f0,
        rate = 20Hz,
        μ = 4.0,
        ), 
)

function soma_network(config)
    @unpack afferents, connections, Npop = config
    E = IF(N=Npop.E, param=config.exc, name="E")
    I = IF(N=Npop.I, param=config.inh, name="I")

    AfferentParam = PoissonStimulusLayer(afferents.rate; afferents...)
    afferentE = PoissonLayer(E, :ge, param=AfferentParam, name="noiseE")
    afferentI = PoissonLayer(I, :ge, param=AfferentParam, name="noiseI")

    synapses = (
        E_to_E = SpikingSynapse(E, E, :ge, p=connections.E_to_E.p, μ=connections.E_to_E.μ, name="E_to_E"),
        E_to_I = SpikingSynapse(E, I, :ge, p=connections.E_to_I.p, μ=connections.E_to_I.μ, name="E_to_I"),
        I_to_E = SpikingSynapse(I, E, :gi, p=connections.I_to_E.p, μ=connections.I_to_E.μ, name="I_to_E"),
        I_to_I = SpikingSynapse(I, I, :gi, p=connections.I_to_I.p, μ=connections.I_to_I.μ, name="I_to_I"),
    )
    model = merge_models(;E,I, afferentE, afferentI, synapses..., silent=true, name="Balanced network") 
    monitor!(model.pop, [:fire])
    monitor!(model.stim, [:fire])
    # monitor!(model.pop, [:v], sr=200Hz)
    return merge_models(;model..., silent=true)
end



νa =  exp.(range(log(1), log(20), 20))
f_rate = map(νa) do x
    frs = tmap(1:5) do _
        config = @update Zerlaut2019_network begin
            afferents.rate = x*Hz
        end 
        model = soma_network(config)
        sim!(;model, duration=10_000ms,  pbar=false)
        fr= firing_rate(model.pop.E, interval=3s:10s, pop_average=true, time_average=true)
    end
        f =     mean(frs)
    @info "rate: $x Hz = $(mean(f))"
    frs
end

ff_rate = [filter(x -> x < 80, mean.(fr)) for fr in f_rate]
scatter(νa, mean.(ff_rate), ribbon=std.(ff_rate), scale=:log10, xlims=(0.9,20), ylims=(0.0000001,80))#, xscale=:log, yscale=:log)
plot!(νa, mean.(ff_rate), ribbon=std.(ff_rate), scale=:log10, xlims=(0.9, 20), ylims=(0.0000001,80), lw=5, xticks=([1,5,10, 20], [1,5,10,20]))#, xscale=:log, yscale=:log)
plot!(xlabel="Afferent rate (Hz)", ylabel="Firing rate (Hz)",  legend=false, size=(400,400))



