# Aggregate scaling (homeostatic scaling of the summed input weight towards a target rate): 100 AdEx neurons
# receive sparse Poisson input; the rule rescales the summed input weight of every neuron until its rate
# matches the target. The target is 1 Hz for 30 s and is then lowered to 0.5 Hz. Top: mean rate and target;
# bottom: mean summed input weight.
# (The earlier version of this script perturbed weights and targets with a random walk at every step; it is
# replaced by this single step of the target.)
# Runtime: about 45 s.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie, Statistics

E = SNN.AdEx(N = 100, name = "E")
P = SNN.Poisson(N = 100, param = SNN.PoissonParameter(3Hz))
S = SNN.SpikingSynapse(P, E, :ge; conn = (μ = 10, σ = 3, p = 0.2))
param = SNN.AggregateScalingParameter(
    τe = 100ms,
    τa = 300ms,
    τ = 10ms,
    Wmax = 400pF,
    Y = fill(Float32(1Hz), E.N),    # target rate of every neuron
)
A = SNN.AggregateScaling(E, [S], param = param)
model = SNN.compose(E = E, P = P, S = S, A = A, silent = true)
SNN.monitor!(E, [:fire], monitor_time = 60s)

chunks = 12
rate = zeros(chunks)
target = zeros(chunks)
sumW = zeros(chunks)
for k = 1:chunks
    k == 7 && (param.Y .= Float32(0.5Hz))        # lower the target after 30 s
    SNN.train!(; model, duration = 5s)
    fr, _ = SNN.firing_rate(E, interval = ((k-1)*5s+1s):100ms:(k*5s), time_average = true)
    rate[k] = mean(fr)
    target[k] = mean(param.Y) / Hz
    sumW[k] = mean(sum(S.W, dims = 2))
end

t = (1:chunks) .* 5
fig = Figure(size = (700, 550))
ax1 = Axis(fig[1, 1], ylabel = "Rate (Hz)")
scatterlines!(ax1, t, rate, label = "mean rate")
stairs!(ax1, t, target, color = :black, label = "target", step = :pre)
axislegend(ax1, position = :rb)
ax2 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Summed input weight")
scatterlines!(ax2, t, sumW, color = :gray)
fig
