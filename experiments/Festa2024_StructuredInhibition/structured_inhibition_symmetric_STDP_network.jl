using DrWatson
using Revise
using Plots
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Statistics
using ProgressBars
# Define the network

network = create_network(;stdp, network_param, neuron_param ) = let
    # Number of neurons in the network
    @unpack Ne, Ni, pee, pei, pie, pii, wee, wei, wii, wie= network_param
    # Create dendrites for each neuron
    E = SNN.IF(N = Ne, param = neuron_param, name="Excitatory")
    I1 = SNN.IF(N = Ni, param = neuron_param, name="Inh1")
    # Define interneurons 
    synapses = Dict{Symbol,Any}(
        :E_to_E => SNN.SpikingSynapse(E, E, :ge, p = pee, μ = wee, name = "E_to_E"),
        :E_to_I1 => SNN.SpikingSynapse(E, I1, :ge, p = pei, μ = wei, name = "E_to_I1"),
        :I1_to_I1 => SNN.SpikingSynapse(I1, I1, :gi, p = pii, μ = wii, name = "I1_to_I1"),
        :I1_to_E => SNN.SpikingSynapse(I1, E, :gi, p = pie, μ = wie, param = stdp, name = "I1_to_E"),
    ) |> dict2ntuple

    pop = dict2ntuple(@strdict E I1)
    stim = Dict{Symbol,Any}(
        :stim_e => SNN.PoissonStimulus(E, :ge, param=50Hz*50, μ=1f0, cells=:ALL),
        :stim_i => SNN.PoissonStimulus(I1, :ge, param=50Hz*40, μ=0.2f0, cells=:ALL),
        # :stim_i => SNN.CurrentStimulus(I1, I_base=I_base, cells=:ALL),
    )
    merge_models(pop, synapses, stim, silent=true)
end



Alearn = 4e-2
stdp_sym = SymmetricSTDP(   A_x = 1e3*Alearn,
                            A_y = 1e3*Alearn,
                            αpre = -0.05*Alearn,
                            αpost = 0.0,
                            τ_x = 30ms,
                            τ_y = 600ms,
                            Wmax = 80,
                           )
# stdp_asym= AntiSymmetricSTDP(A_x = 5e-3,
#                             A_y = 2e-2,
#                             αpre = -0.1,
#                             αpost = 0.0,
#                             τ_x = 60ms,
#                             τ_y = 30ms,
#                             Wmax = 80
#                            )

# SNN.stdp_integral(stdp_sym, fill=false, ΔTs = -700:5:700ms)
# SNN.stdp_kernel(stdp_sym, fill=false, ΔTs = -500:10:500ms)

C = 200pF
gl = 10nS
neuron_param = IFParameterSingleExponential(
        R = 1/ gl,
        τm= C/gl,
        τabs = 5ms,
        Vt = -50mV,
        El=-60mV, 
        # synapses
        E_e = 0mV,
        E_i = -80mV,
        τe= 5ms,
        τi= 10ms,
        )


network_param =(
        Ne = 800,
        Ni = 200,
        pee = 0.2,
        pei = 0.1,
        pie = 0.3,
        pii = 0.2,
        wee = 1.0,
        wei = 1.0,
        wie = 1.8,
        wii = 0.3,
        )

model= create_network(;stdp=stdp_sym, network_param, neuron_param)
#

@info "Initializing network"
simtime = SNN.Time()
monitor(model.pop, [:fire])
monitor(model.syn, [:W],  sr=1Hz)
monitor(model.pop, [:gi], sr=20Hz)
monitor(model.pop, [:ge], sr=20Hz)
monitor(model.pop, [:v],  sr=20Hz)
sim!(model=model, duration = 50s, time = simtime, pbar = true)
##
duration=300 #seconds 
rI, rE, W, τrate = 0, 0, 0, 150ms
iter = ProgressBar(1:duration)
for x in iter
    for i in 1:10
        train!(model=model, duration = 0.1s, time = simtime, dt = dt, pbar = false)
        W = mean(model.syn.I1_to_E.W)
        rI += mean(model.pop.I1.fire) - rI/τrate
        rE += mean(model.pop.E.fire)  - rE/τrate
    end
    set_multiline_postfix(iter, "Inh W: $(W)\nE rate: $(rE/dt/Hz/τrate)\nI rate: $(rI/dt/Hz/τrate)")
end
path = datadir("structured_inhibition", "asymmetric") |> mkpath
save_model(path=path, model=model, name="structured_symmetric", simtime=simtime)


## Loade model and plot
data = load_data(path=path, name="structured_symmetric", info=true)
T = get_time(simtime)
@unpack model = data
p = plot_network_plasticity(model, simtime, ΔT= 4s, interval = T-120s:1s:T)

mutual_EI_connections(synapses)

