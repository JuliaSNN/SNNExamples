using DrWatson
# @quickactivate "network_models"
using Plots
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Random, Statistics, StatsBase
using Statistics, SparseArrays
using StatsPlots
using ProgressBars

# %% [markdown]
# Network

Random.seed!(123)
network = let
    NE = 400
    NI = 100
    E = SNN.TripodHet(
        N = NE,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
        dend_syn = Synapse(EyalGluDend, MilesGabaDend),
        NMDA = SNN.EyalNMDA,
        param = SNN.AdExSoma(Vr = -50),
    )
    I1 = SNN.IF(; N = NI ÷ 2, param = SNN.IFParameter(τm = 7ms, El = -55mV))
    I2 = SNN.IF(; N = NI ÷ 2, param = SNN.IFParameter(τm = 20ms, El = -55mV))
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge, p = 0.2, μ = 15.0)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge, p = 0.2, μ = 15.0)
    I2_to_E = SNN.CompartmentSynapse(
        I2,
        E,
        :d1,
        :inh,
        p = 0.2,
        μ = 5.0,
        param = SNN.iSTDPParameterPotential(v0 = -60mV),
    )
    I1_to_E = SNN.CompartmentSynapse(
        I1,
        E,
        :s,
        :inh,
        p = 0.2,
        μ = 5.0,
        param = SNN.iSTDPParameterRate(r = 4Hz),
    )
    E_to_E_d1 = SNN.CompartmentSynapse(
        E,
        E,
        :d1,
        :exc,
        p = 0.2,
        μ = 10,
        param = SNN.vSTDPParameter(),
    )
    E_to_E_d2 = SNN.CompartmentSynapse(
        E,
        E,
        :d2,
        :exc,
        p = 0.2,
        μ = 10,
        param = SNN.vSTDPParameter(),
    )
    recurrent_norm_d1 = SNN.SynapseNormalization(
        E,
        [E_to_E_d1],
        param = SNN.MultiplicativeNorm(τ = 100ms),
    )
    recurrent_norm_d2 = SNN.SynapseNormalization(
        E,
        [E_to_E_d2],
        param = SNN.MultiplicativeNorm(τ = 100ms),
    )
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_E_d1 E_to_E_d2 I1_to_E I2_to_E E_to_I1 E_to_I2 norm1=recurrent_norm_d1 norm2=recurrent_norm_d2)
    (pop = pop, syn = syn)
end

function ramp(time::Float32)
    if time > 5500ms && time < 5700ms
        return 1000Hz
    else
        return 0.0
    end
end

function get_rand_cells(pop, L)
    N = pop.N
    return rand(1:N, L)
end

get_rand_cells(network.pop.E, 10)

stimuli = Dict(
    ## Background noise
    "noise_s"  => SNN.PoissonStimulus(network.pop.E, :h_s, x->1000Hz, cells=:ALL, μ=10.f0),
    "stim1_d1" => SNN.PoissonStimulus(network.pop.E, :h_d1, ramp, μ=10.f0, cells=get_rand_cells(network.pop.E, 10)),
)


model = SNN.merge_models(network, stim=stimuli)

##
timer= SNN.Time()
SNN.monitor(network.pop.E, [:v_d1, :v_s, :fire, :h_s])
SNN.monitor(network.pop.I1, [:fire])
SNN.monitor(network.pop.I2, [:fire])
SNN.clear_records([model.pop...])
SNN.train!(model=model, duration = 5000ms, pbar = true, dt = 0.125, time=timer)


##
SNN.train!(model=model, duration = 1000ms, dt = 0.125, time=timer)
SNN.raster([network.pop...], (5000,6000))
SNN.raster([network.pop.E], (5000,6000); neurons=[stimuli["stim1_d1"].cells, stimuli["stim1_d1"].cells])

fr, interval = SNN.firing_rate(network.pop.E, interval=5000:10:6000)
average = SNN.average_firing_rate(network.pop.E)
st = SNN.spiketimes(network.pop.E)

st[1]
plot(interval, fr)
scatter(average)


SNN.vecplot(network.pop.E, :v_d1, r=5400:6000, neurons=cells, dt=0.125)
##