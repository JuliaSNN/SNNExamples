# Persistent activity in two coupled E/I assemblies with different inhibitory plasticity rules on I -> E:
# rate-based iSTDP, Hebbian and anti-Hebbian anti-symmetric STDP, and symmetric STDP. Each network is warmed up
# for 30 s, assembly 1 receives a Poisson cue between 3 and 4 s of the test phase (10 s), and the I -> E weights,
# the rates and a raster are plotted (one figure per rule, saved to plots/iSTDP_lateralExc/).
# The original also had `iSTDPTime`, which no longer exists; it is not part of the comparison.
# Runtime: about 9.5 min.
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie
using Statistics
include("models.jl")

attractor = 5
Alearn = 4e-3
Ebase = 0.1
C = 200pF
gl = 10nS
R = 1 / gl
τm = C / gl

config = (
    intervals = [(:sub_1_E, [3s, 4s])],
    E_to_I = (p = 0.2, μ = 1.0),
    E_to_E = (p = 0.2, μ = attractor * Ebase),
    I_to_I = (p = 1.0, μ = 0.3),
    I_to_E = (p = 1.0, μ = 2.0),
    lateral_EI = (p = 0.2, μ = 2.5),
    lateral_EE = (p = 0.2, μ = Ebase),
    N = 500,
    n_assemblies = 2,
    duration = 10s,
    adex = SNN.AdExParameter(; R = R, τm = τm, Vr = -55mV, a = 0nS, b = 0pA),
    path = mkpath(plotsdir("iSTDP_lateralExc")),
    noise = 2.5kHz,
    ext_stim = 0.8kHz,
    warmup = 30s,
)
istdps = (
    istdp_rate = SNN.iSTDPRate(τy = 20ms, η = Alearn * 1e3, r = 5Hz),
    Hebbian = SNN.STDPAntiSymmetric(A_x = Alearn * 1e3, A_y = 0.7Alearn * 1e3, τ_x = 60ms, τ_y = 30ms, Wmax = 200),
    antiHebbian = SNN.STDPAntiSymmetric(A_x = -Alearn * 1e3, A_y = -0.7Alearn * 1e3, τ_x = 60ms, τ_y = 30ms, Wmax = 200),
    Symmetric = SNN.STDPSymmetric(A_x = Alearn * 1e3, A_y = Alearn * 1e3, αpre = -0.5, αpost = 0.0,
                                  τ_x = 30ms, τ_y = 600ms, Wmax = 200),
)

models = Dict()
for name in keys(istdps)
    istdp = istdps[name]
    local_config = (config..., istdp = istdp)
    model = test_istdp(local_config)
    fr, r = SNN.firing_rate(model.pop.sub_1_E, interval = 1s:10ms:3s, time_average = true)
    @info "Rule $name, firing rate of assembly 1 before the cue: $(mean(fr)) Hz"
    fig = iSTDP_activity(model, istdp, local_config)
    save(joinpath(config.path, "iSTDP_$(name).png"), fig)
    push!(models, name => model)
end
