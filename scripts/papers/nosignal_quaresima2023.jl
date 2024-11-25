using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

## Define the network parameters

network = let
    # Number of neurons in the network
    NE = 1500
    NI = NE ÷ 4
    NI1 = round(Int,NI * 0.35)
    NI2 = round(Int,NI * 0.65)

    # Import models parameters
    I1_params = duarte2019.PV
    I2_params = duarte2019.SST
    @unpack connectivity, plasticity = quaresima2023
    @unpack dends, NMDA, param, soma_syn, dend_syn = quaresima2022

    # Define interneurons I1 and I2
    I1 = SNN.IF(; N = NI1, param = I1_params, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = I2_params, name="I2_sst")
    E = SNN.BallAndStick(250um; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="Exc")
    # background noise
    noise = Dict(
        # :noise_s   => SNN.PoissonStimulus(E,  :he_s,  param=1.0kHz, cells=:ALL, μ=0.f0, name="noise_s",),
        :d   => SNN.PoissonStimulus(E,  :he_d,  param=2.0kHz, cells=:ALL, μ=1.f0, name="noise_s",),
        :i1  => SNN.PoissonStimulus(I1, :ge,   param=1.5kHz, cells=:ALL, μ=1.f0,  name="noise_i1"),
        :i2  => SNN.PoissonStimulus(I2, :ge,   param=2.0kHz, cells=:ALL, μ=1.8f0, name="noise_i2")
    )
    syn= Dict(
    :I1_to_I1 => SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...),
    :I1_to_I2 => SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...),
    :I2_to_I2 => SNN.SpikingSynapse(I1, I2, :gi; connectivity.IsIs...),
    :I2_to_I1 => SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...),
    :I1_to_E  => SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.EIf...),
    :I2_to_E  => SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.EdIs...),
    :E_to_I1  => SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...),
    :E_to_I2  => SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...),
    :E_to_E   => SNN.SpikingSynapse(E, E, :he, :d ; connectivity.EdE...),
    )
    pop = dict2ntuple(@strdict I1 I2 E)

    # Return the network as a model
    merge_models(pop, noise=noise, syn)
end
 

# %%
# Run the model
# %%
## Target activation with stimuli

mytime = SNN.Time()
SNN.monitor([network.pop...], [:fire, :v_d, :v_s, :v, (:g_d, [10,20,30,40,50]), (:ge_s, [10,20,30,40,50]), (:gi_s, [10,20,30,40,50])], sr=200Hz)
SNN.train!(model=network, duration= 10s, pbar=true, dt=0.125, time=mytime)
T = get_time(mytime)
Trange = 1s:10ms:T

##

# SNN.vecplot( network.pop.E, :, r = Trange, neurons = 1:100, pop_average=true, label="soma")

    
function plot_activity(network, Trange)
    frE, interval = SNN.firing_rate(network.pop.E, interval = Trange)
    frI1, interval = SNN.firing_rate(network.pop.I1, interval = Trange)
    frI2, interval = SNN.firing_rate(network.pop.I2, interval = Trange)
    pr = plot(xlabel = "Time (ms)", ylabel = "Firing rate (Hz)")
    plot!(Trange, mean(frE), label = "E", c = :black)
    plot!(Trange, mean(frI1), label = "I1", c = :red)
    plot!( Trange,mean(frI2), label = "I2", c = :green)
    plot!(margin = 5Plots.mm, xlabel="")
    pv =SNN.vecplot(network.pop.E, :v_d, r = Trange,  pop_average = true, label="dendrite")
    SNN.vecplot!(pv, network.pop.E, :v_s, r = Trange, pop_average = true, label="soma")
    plot!(ylims=:auto, margin = 5Plots.mm, ylabel = "Membrane potential (mV)", legend=true, xlabel="")
    dgplot = dendrite_gplot(network.pop.E, :d, r=Trange, dt=0.125, margin=5Plots.mm, xlabel="")
    sgplot = soma_gplot(network.pop.E, :s, r=Trange, dt=0.125, margin=5Plots.mm, xlabel="")
    rplot = SNN.raster(network.pop, Trange, size=(900,500), margin=5Plots.mm, xlabel="")
    layout = @layout  [ 
                c{0.2h}
                e{0.2h}
                a{0.2h}
                b{0.2h}
                d{0.2h}]
    plot(pr, pv, rplot, dgplot, sgplot, layout=layout, size=(900, 1200), topmargn=0Plots.mm, bottommargin=0Plots.mm, bgcolorlegend=:transparent, fgcolorlegend=:transparent)
end
plot_activity(network, 1:10ms:10s)
##
W = network.syn.I1_to_E.W
h_I1E = histogram(W, bins=minimum(W):maximum(W)/200:maximum(W)+1, title = "Synaptic weights from I1 to E",  xlabel="Synaptic weight", ylabel="Number of synapses", yticks=:none, c=:black)
W = network.syn.I2_to_E.W
h_I2E = histogram(W, bins=minimum(W):maximum(W)/200:maximum(W)+1, title = "Synaptic weights from I2 to E", xlabel="Synaptic weight", ylabel="Number of synapses", yticks=:none, c=:black)
sc_w = scatter(network.syn.I2_to_E.W, network.syn.I1_to_E.W,  xlabel="Synaptic weight from I2 to E", ylabel="Synaptic weight from I1 to E", alpha=0.01, c=:black)
frE= SNN.average_firing_rate(network.pop.E, interval = Trange)
sc_fr=histogram(frE, c=:black, label="E", xlabel="Firing rate (Hz)",bins=-0.5:0.2:12, ylabel="Number of neurons")
layout = @layout  [ 
            grid(2,2)
            ]
plot(h_I2E, h_I1E, sc_w, sc_fr, layout=layout, size=(800, 600), legend=false, margin=5Plots.mm)
## 

