# Response of a Tripod neuron to one excitatory and one inhibitory spike delivered to the soma, with the
# default soma and dendrite receptors (Duarte AMPA and Miles GABA_A on the soma, SNNModels >= 2.1 defaults).
# The neuron rests for 1 s; at 1050 ms one spike of weight 20 reaches the glutamatergic and the GABAergic
# soma synapse together. Plots the somatic and dendritic voltage, the adaptation current and the somatic
# receptor conductances.
# Runtime: about 45 s.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie

E = SNN.Tripod(N = 1, adex = SNN.AdExParameter(El = -55mV))

param = SNN.SpikeTimeParameter([1050ms], [1])
w = fill(20.0f0, 1, 1)
exc_soma = SNN.SpikeTimeStimulus(E, :glu, :s; param, conn = w)
inh_soma = SNN.SpikeTimeStimulus(E, :gaba, :s; param, conn = w)
model = SNN.compose(; E, exc_soma, inh_soma, silent = true)

SNN.monitor!(E, [:v_s, :v_d1, :w_s, :fire], sr = 1000Hz)
SNN.monitor!(E, :g, variables = :synvars_s, sr = 1000Hz)
SNN.sim!(model, 1250ms)

interval = 1000ms:0.1ms:1250ms
fig = Figure(size = (700, 800))
ax1 = Axis(fig[1, 1], ylabel = "V soma (mV)")
SNN.vecplot!(ax1, E, :v_s; interval, add_spikes = true)
ax2 = Axis(fig[2, 1], ylabel = "V dendrite 1 (mV)")
SNN.vecplot!(ax2, E, :v_d1; interval)
ax3 = Axis(fig[3, 1], ylabel = "w soma (pA)")
SNN.vecplot!(ax3, E, :w_s; interval)
ax4 = Axis(fig[4, 1], ylabel = "g soma (nS)", xlabel = "Time (s)")
for (i, label) in enumerate(["glu", "gaba"])
    SNN.vecplot!(ax4, E, :g; variables = :synvars_s, sym_id = i, interval, neurons = 1, label, color = [:red, :blue][i])
end
axislegend(ax4)
fig
