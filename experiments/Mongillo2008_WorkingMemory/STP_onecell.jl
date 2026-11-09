# Short-term plasticity of a single synapse with the parameters of Mongillo et al. (2008): a presynaptic
# neuron fires six spikes at 33 Hz, then one spike after a pause of 200 ms; the efficacy variables x (resources)
# and u (utilisation) and the postsynaptic voltage are plotted.
# Runtime: about 35 s.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie
include("mongillo_WM2008.jl")

# presynaptic spikes: 6 spikes every 30 ms starting at 5 s, one more 200 ms after the last
spike_times = vcat([5s + (k - 1) * 30ms for k = 1:6], 5s + 5 * 30ms + 200ms)
pre = SNN.Identity(N = 1)
drive = SNN.SpikeTimeStimulusIdentity(pre, :g; param = SNN.SpikeTimeParameter(spike_times, ones(Int, length(spike_times))))
post = SNN.IF(N = 1, param = MongilloParam.Exc, spike = MongilloParam.spike, synapse = MongilloParam.synapse)
EE = SNN.SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, 1, 1), STPParam = SNN.MarkramSTPParameter(τD = 200ms, τF = 1500ms, U = 0.2))
SNN.monitor!(post, [:v], sr = 250Hz)
SNN.monitor!(EE, [:u, :x], variables = :STPVars, sr = 250Hz)
model = SNN.compose(; pre, post, EE, drive, silent = true)
SNN.train!(model = model, duration = 15s, dt = 0.125f0)

r = 4.9s:4ms:6s
x = SNN.record(EE, :STPVars_x)
u = SNN.record(EE, :STPVars_u)
v = SNN.record(post, :v)
fig = Figure(size = (700, 500))
ax1 = Axis(fig[1, 1], ylabel = "x, u", title = "Tsodyks-Markram synapse")
lines!(ax1, r ./ 1000, x(1, r), color = :red, label = "x")
lines!(ax1, r ./ 1000, u(1, r), color = :blue, label = "u")
ylims!(ax1, 0, 1)
axislegend(ax1, position = :rt)
ax2 = Axis(fig[2, 1], ylabel = "V (mV)", xlabel = "Time (s)")
lines!(ax2, r ./ 1000, v(1, r), color = :black)
fig
