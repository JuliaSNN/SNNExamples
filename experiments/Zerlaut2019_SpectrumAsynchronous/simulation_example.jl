using DrWatson
findproject() |> quickactivate
using UnPack
using SpikingNeuralNetworks
import SNNPlots: Plots, plot, raster
SNN.@load_units
##

Zerlaut2019_network = (Npop = (E=8000, I=2000),
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

function network(config)
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


##
plots = map([4, 10]) do input_rate
    config = @update Zerlaut2019_network begin
        afferents.rate = input_rate*Hz
    end 
    model = soma_network(config)
    sim!(;model, duration=10_000ms,  pbar=true)
    pr= raster(model.pop, every=40)

    # Firing rate of the network with a fixed afferent rate
    frE, r = firing_rate(model.pop.E, interval=3s:10s, pop_average=true)
    frI, r = firing_rate(model.pop.I, interval=3s:10s, pop_average=true)
    pf = plot(r, [frE, frI], labels=["E" "I"],
        xlabel="Time (s)", ylabel="Firing rate (Hz)", 
        title="Afferent rate: $input_rate Hz",
        size=(600, 400), lw=2)

    # Plot the raster plot of the network
    plot(pf, pr, layout=(2, 1))
end

p = plot(plots..., layout=(1,2), size=(1200, 600), xlabel="Time (s)", leftmargin=10Plots.mm)
##

savefig(p, "/home/user/mnt/zeus/User_folders/aquaresi/network_models/src/SpikingNeuralNetworks.jl/docs/src/assets/examples/recurrent_network.png")