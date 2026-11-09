# Structured inhibition (Festa et al. 2024): 900 excitatory and 100 inhibitory IF neurons with a plastic I -> E
# connection that follows an ASYMMETRIC (anti-symmetric) STDP rule (STDPAntiSymmetric) and Poisson drive. After
# 50 s of initial simulation the network is trained for another 600 s; the script then plots the mean I -> E weight,
# the E and I rates, a raster of the last seconds and the weights of the reciprocal E <-> I pairs.
# Only the I -> E weights are recorded (the original recorded all weights, which is several hundred MB).
# Runtime: about 1.5 min.
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie
using Statistics

## Analysis helpers (the originals, `plot_network_plasticity` and `mutual_EI_connections`, were part of SNNUtils < 1)
# I -> E weight over time, and E/I rates
function plot_network_plasticity(model; interval)
    fig = Figure(size = (800, 700))
    ax1 = Axis(fig[1, 1], ylabel = "Mean I → E weight", xlabel = "Time (s)")
    W, r = SNN.record(model.syn.I1_to_E, :W, range = true)
    wmean = vec(mean(SNN.getvariable(model.syn.I1_to_E, :W), dims = 1))
    lines!(ax1, r ./ 1000, wmean, linewidth = 2)
    ax2 = Axis(fig[2, 1], ylabel = "Rate (Hz)", xlabel = "Time (s)")
    for p in (model.pop.E, model.pop.I1)
        fr, rr = SNN.firing_rate(p; interval, τ = 100ms, pop_average = true)
        lines!(ax2, rr ./ 1000, vec(fr), label = p.name)
    end
    axislegend(ax2)
    ax3 = Axis(fig[3, 1], ylabel = "Neuron", xlabel = "Time (s)")
    SNN.raster!(ax3, model.pop, (last(interval) - 5s):last(interval), every = 2)
    return fig
end

# I -> E weights of the pairs that are also connected E -> I (reciprocal pairs) compared with the other I -> E pairs
function mutual_EI_connections(syn)
    W_EI = Matrix(SNN.matrix(syn.E_to_I1))      # N_I x N_E
    W_IE = Matrix(SNN.matrix(syn.I1_to_E))      # N_E x N_I
    reciprocal = (W_EI .> 0)' .& (W_IE .> 0)    # N_E x N_I, connected in both directions
    one_way = (W_EI .== 0)' .& (W_IE .> 0)
    fig = Figure(size = (500, 400))
    ax = Axis(fig[1, 1], xlabel = "I → E weight", ylabel = "Number of synapses",
              title = "$(count(reciprocal)) reciprocal and $(count(one_way)) one-way I → E pairs")
    hist!(ax, W_IE[one_way], bins = 40, color = (:gray, 0.6), label = "I → E only")
    hist!(ax, W_IE[reciprocal], bins = 40, color = (:red, 0.5), label = "reciprocal")
    axislegend(ax)
    return fig
end

## Define the network
network_param = (
    Ne = 900,
    Ni = 100,
    pee = 0.2,
    pei = 0.1,
    pie = 0.1,
    pii = 0.2,
    wee = 1.0,
    wei = 1.0,
    wie = 1.8,
    wii = 0.3,
)

function create_network(; stdp, network_param, neuron_param, synapse)
    (; Ne, Ni, pee, pei, pie, pii, wee, wei, wii, wie) = network_param
    E = SNN.IF(N = Ne, param = neuron_param, synapse = synapse, spike = SNN.PostSpike(τabs = 5ms), name = "Excitatory")
    I1 = SNN.IF(N = Ni, param = neuron_param, synapse = synapse, spike = SNN.PostSpike(τabs = 5ms), name = "Inh1")

    E_to_E = SNN.SpikingSynapse(E, E, :ge; conn = (p = pee, μ = wee), name = "E_to_E")
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; conn = (p = pei, μ = wei), name = "E_to_I1")
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; conn = (p = pii, μ = wii), name = "I1_to_I1")
    I1_to_E = SNN.SpikingSynapse(I1, E, :gi; conn = (p = pie, μ = wie), LTPParam = stdp, name = "I1_to_E")

    stim_e = SNN.PoissonStimulus(E, :ge; param = SNN.PoissonFixed(rate = 50Hz * 40, μ = 1.0f0), neurons = :ALL)
    stim_i = SNN.PoissonStimulus(I1, :ge; param = SNN.PoissonFixed(rate = 50Hz * 40, μ = 0.2f0), neurons = :ALL)
    return SNN.compose(; E, I1, E_to_E, E_to_I1, I1_to_I1, I1_to_E, stim_e, stim_i, silent = true)
end

Alearn = 5e-3
stdp = SNN.STDPAntiSymmetric(
    A_x = Alearn * 1e3,
    A_y = 0.7 * Alearn * 1e3,
    αpre = -0.7 * Alearn,
    αpost = 0.2 * Alearn,
    τ_x = 60ms,
    τ_y = 30ms,
    Wmax = 80,
)

C = 200pF
gl = 10nS
neuron_param = SNN.IFParameter(R = 1 / gl, τm = C / gl, Vt = -50mV, El = -60mV)
synapse = SNN.SingleExpSynapse(E_e = 0mV, E_i = -80mV, τe = 5ms, τi = 10ms)

model = create_network(; stdp, network_param, neuron_param, synapse)

## Initial simulation and training
@info "Initializing network"
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(model.syn.I1_to_E, [:W], sr = 1Hz)
SNN.train!(model = model, duration = 50s, dt = 0.125f0)
for x = 1:6
    SNN.train!(model = model, duration = 100s, dt = 0.125f0)
    @info "t = $(round(SNN.get_time(model) / 1000)) s, mean I → E weight = $(mean(model.syn.I1_to_E.W))"
end

## Save the model and plot (recordings are not kept in a saved model, so the plots use the model in memory)
path = datadir("structured_inhibition", "asymmetric")
mkpath(path)
info = (name = "structured_asymmetric",)
SNN.save_model(; model, path, name = "structured_asymmetric", info, config = info)
T = SNN.get_time(model)
plot_network_plasticity(model; interval = (T-20s):100ms:T)
mutual_EI_connections(model.syn)
