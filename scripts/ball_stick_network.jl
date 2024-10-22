# Import necessary packages
using DrWatson
using Plots
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Random, Statistics, StatsBase
using Statistics, SparseArrays
using StatsPlots
using ProgressBars


# Define the network
network = let
    # Number of neurons in the network
    N = 1000
    NI = N ÷ 8
    # Create dendrites for each neuron
    # Define neurons and synapses in the network
    E = SNN.BallAndStick(
        (150um, 200um);
        N = N,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
        dend_syn = Synapse(EyalGluDend, MilesGabaDend),
        NMDA = SNN.EyalNMDA,
        param = SNN.AdExSoma(b = 0.0f0, Vr = -50),
    )
    # Define interneurons I1 and I2
    I1 =
        SNN.IF(; N = NI, param = SNN.IFParameterSingleExponential(τm = 7ms, Vr = -52mV))
    I2 = SNN.IF(; N = NI, param = SNN.IFParameterSingleExponential(τm = 20ms, Vr = -52mV))
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge, p = 0.2, μ = 4.50)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge, p = 0.2, μ = 4.50)

    I2_to_E = SNN.CompartmentSynapse(
        I2,
        E,
        :d,
        :inh,
        p = 0.2,
        μ = 1,
        param = SNN.iSTDPParameterPotential(v0 = -65mV),
    )
    I1_to_E = SNN.CompartmentSynapse(
        I1,
        E,
        :s,
        :inh,
        p = 0.2,
        μ = 1,
        param = SNN.iSTDPParameterRate(r = 10Hz),

    )
    E_to_E = SNN.CompartmentSynapse(
        E,
        E,
        :d,
        :exc,
        p = 0.2,
        μ = 10.0,#
    )
    recurrent_norm =
        SNN.SynapseNormalization(N, [E_to_E], param = SNN.AdditiveNorm(τ = 10ms))
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_E I1_to_E I2_to_E E_to_I1 E_to_I2 norm=recurrent_norm)
    # Return the network as a tuple
    (pop = pop, syn = syn)
end

#
stimuli = Dict(
    "noise_s"  => SNN.PoissonStimulus(network.pop.E, :h_s, x->4000Hz, cells=:ALL, μ=4.7f0),
    "noise_i1"  => SNN.PoissonStimulus(network.pop.I1, :ge, x->30Hz, cells=:ALL, μ=1.f0),
    "noise_i2"  => SNN.PoissonStimulus(network.pop.I2, :ge, x->30Hz, cells=:ALL, μ=1.f0),
)
model = SNN.merge_models(network, stim=stimuli)
SNN.monitor(model.pop.E, [:v_s, :v_d, :h_s, :h_d])
SNN.monitor([model.pop...], [:fire])
SNN.train!(model=model, duration= 10s, pbar=true, dt=0.125)

##
plot(
    SNN.raster([network.pop...], (9000ms, 10000ms)),
    SNN.raster([network.pop...], (4000ms, 5000ms)),
)

SNN.vecplot(network.pop.E, :v_d, neurons = 1:1, r = 1.001s:8.9s, dt=0.125)
SNN.vecplot(network.pop.E, :h_d, neurons = 1:1, r = 1.8s:0.001:7.9s, sym_id = 1, dt=0.125)
histogram(network.syn.E_to_E.W)

simulation_time = 60
W = zeros(simulation_time, 3)
for t = 1:simulation_time
    train!(populations, synapses, duration = 1s)
    W[t, 1] = mean(network.syn.E_to_E.W)
    W[t, 2] = mean(network.syn.I1_to_E.W)
    W[t, 3] = mean(network.syn.I2_to_E.W)
end

plot(W)

##
SNN.monitor([network.pop...], [:fire])
SNN.monitor([network.pop.E], [:v_d])
train!(populations, synapses, duration = 5s)
SNN.raster([network.pop...], (4000ms, 5000ms))
##


# # %%
# # Setting up the network monitoring to track neural activity during simulation
# SNN.monitor([network.pop...], [:fire])
# SNN.monitor(network.pop.E, [:v_s, :v_d1])
# # Training the network populations and synapses for a duration of 10 seconds
# @info "Initializing network"
# train!(populations, synapses, duration = 6s)
# ##
# # sim!(populations, synapses, duration = 8000ms)
# # Displaying a raster plot of the network activity between 5 and 10 seconds into the simulation

network.syn.I2_to_E.W

# s=1000
# SNN.vecplot!(plot(),network.pop.E, :v_s, r=4s:5.5s; neurons=1:10)
# SNN.vecplot!(plot(),network.pop.E, :v_d1, r=4s:5.5s; neurons=1:10)

# # %% [markdown]
# #

# %% Run the training of the weights
# Another round of training followed by clearing the records of network activity and setting up monitoring
SNN.clear_records([network.pop...])
SNN.monitor([network.pop...], [:fire])
SNN.monitor([network.pop.E], [:fire, :v_s, :v_d1])

# Setting up a loop for repeated training simulations where input rate is adjusted before each run.
# Weights are recorded after each run.
repetitions = 1:70
learn_int = 2 * length(repetitions) ÷ 3
W = zeros(repetitions, length(inputs.syn))

input_rate = 500Hz
@info "Run training E/I weights"
for x in ProgressBar(repetitions[1:learn_int])
    W[x, :] = [mean(syn.W) for syn in inputs.syn]
    train!(populations, synapses, duration = 200ms)
    for x ∈ 1:2
        set_input_rate!(inputs.pop, x, input_rate)
        train!(populations, synapses, duration = 50ms)
    end
    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 500ms)
end

set_input_rate!(inputs.pop, 0, 0Hz)
for x in ProgressBar(repetitions[(learn_int+1):end])
    train!(populations, synapses, duration = 500ms)
    W[x, :] = [mean(syn.W) for syn in inputs.syn]
end

# Plotting the weights across different epochs

labels = ["1" "2" "3" "Inactive"]
plot_W = plot(
    W,
    label = labels,
    xlabel = "Epochs",
    ylabel = "Average syn. weight (nF)",
    legendtitle = "Input",
    legend = :outerright,
)
##
path =
    plotsdir("sequence_detection", "recurrent_connection", "weights_training.pdf") |>
    x -> (mkpath(dirname(x)); x)
savefig(plot_W, path)

##
# targets = Vector{Vector{Int}}()
# weights = Vector{Vector{Float32}}()
# for syn in inputs.syn
# 	t = [n for n in indexin(syn.I) if !isnothing(n)]
# 	w = syn.W[t]
# 	push!(targets,t)
# 	push!(weights,w)
# end

# violin(weights)
# scatter!(mean.(weights))

# %%


s = 1000
# Plotting the soma potential

p1 = SNN.raster([network.pop...], (10s, 15s))
p2 = SNN.vecplot(network.pop.E, :v_s, neurons = 1:5:20, r = 10s:15s, label = "Soma")

p3 = plot(legend = :topleft)
for i ∈ 1:2
    c = SNNPlots.mpi_palette[i]
    ni = collect(Set(inputs.syn[i].I))[1:1:end]
    p3 = SNN.vecplot!(
        p3,
        network.pop.E,
        :v_d1,
        neurons = ni,
        average = true,
        r = 30s:33s,
        label = labels[i],
        c = c,
    )
end
p3 = plot!(legend = :topleft, ylims = (-80, -10))

p23 = plot(p2, p3, layout = (2, 1))
plot(p1, p23, layout = (1, 2), size = (900, 600))



# %%

##
# Getting mean and standard deviation of inhibitory synapse weights before training
μ0_inh_d1 = mean(back_syn.inh_d1.W)
μ0_inh_d1 = std(back_syn.inh_d1.W)

# Clearing network records, setting background firing rate, and running another training simulation
SNN.clear_records([network.pop...])
back_pop.E.rate .= 70Hz
train!(populations, synapses, duration = 5000ms)
# Displaying a raster plot of the network activity in a given timeframe
p1 = SNN.raster([network.pop...], (0_000ms, 5_000ms))

# Getting mean and standard deviation of inhibitory synapse weights after training
μ_inh_d1 = mean(back_syn.inh_d1.W)
μ_inh_d1 = std(back_syn.inh_d1.W)

# Displaying a bar plot comparing weights of inhibitory synapses before and after training
labels = ["Before Training", "After Training"]
inh_d1_means = [μ0_inh_d1, μ_inh_d1]
inh_d1_stds = [μ0_inh_d1, μ_inh_d1]

bar(
    labels,
    inh_d1_means,
    yerr = inh_d1_stds,
    label = "inh_d1 weights",
    title = "Comparing Weights Before and After Training",
    xlabel = "Training Status",
    ylabel = "Weights",
    legend = :topleft,
)

# %%
samples = 1:30
r_output = zeros(2, length(samples), network.pop.E.N)
pbar = ProgressBar(samples)

# noi.Es.rate .= 20Hz
input_rate = 1000Hz
# Loop to simulate and record firing rates under different conditions
@info "Start of simulation loop for two strings"

SNN.clear_monitor(network.pop.E)
SNN.monitor([network.pop...], [:fire])
for x in samples
    # Simulate for 500ms without input
    sim!(populations, synapses, duration = 500ms)
    set_input_rate!(inputs.pop, 0, 0Hz)
    SNN.clear_records(network.pop.E, :fire)
    for x in [2, 3, 1]
        set_input_rate!(inputs.pop, x, input_rate)
        sim!(populations, synapses, duration = 50ms)
    end
    # Simulating for 150ms post input
    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 150ms)
    r_output[1, x, :] = sum(network.pop.E.records[:fire])
    r = sum(sum(network.pop.E.records[:fire]))
    @info "A B C: $r"
    # Resetting input rates to zero
    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 500ms)
    SNN.clear_records(network.pop.E, :fire)
    for x in [1, 2, 3]
        set_input_rate!(inputs.pop, x, input_rate)
        sim!(populations, synapses, duration = 50ms)
    end
    set_input_rate!(inputs.pop, 0, 0Hz)
    # Simulating for 150ms post input
    sim!(populations, synapses, duration = 150ms)
    # Recording sum of firings
    r_output[2, x, :] = sum(network.pop.E.records[:fire])
    r = sum(sum(network.pop.E.records[:fire]))
    @info "C B A: $r"
    # Resetting input rates to zero
    set_input_rate!(inputs.pop, 0, 0Hz)
end
##



scatter([mean(x.W) for x in inputs.syn])
# %%
# Calculating means and plotting

targets = collect(intersect([Set(syn.I) for syn in inputs.syn[1:2]]...))
sample_aver = mean(r_output, dims = (2))[:, 1, :]
most_active = sort(1:network.pop.E.N, by = x -> sample_aver[2, x], rev = true)[targets]


p1 = scatter(sample_aver[1, :], sample_aver[2, :], label = "All neurons", c = :grey)
p1 = scatter!(
    sample_aver[1, most_active],
    sample_aver[2, most_active],
    label = "Most active",
    c = :orange,
)
scatter!(
    [mean(sample_aver[1, most_active])],
    [mean(sample_aver[2, most_active])],
    c = :black,
    shape = :x,
    ms = 10,
    label = "",
)
lims = [minimum(sample_aver), maximum(sample_aver)]
plot!(lims, lims, ls = :dash, lc = :grey, label = "", legend = :bottomright)
plot!(title = "Average neuron activity")

neuron_average = mean(r_output[:, :, :], dims = 3)[:, :, 1]
correct = neuron_average[1, :]
false_positive = neuron_average[2, :]
p2 = scatter(
    correct,
    false_positive,
    label = "All neurons",
    xlabel = "Input ABC (Hz)",
    ylabel = "Input CBA",
    legend = :bottomright,
    c = :grey,
)

neuron_average = mean(r_output[:, :, most_active], dims = 3)[:, :, 1]
correct = neuron_average[1, :]
false_positive = neuron_average[2, :]
p2 = scatter!(
    correct,
    false_positive,
    label = "Most active",
    xlabel = "Input CBA (Hz)",
    ylabel = "Input ABC (Hz)",
    legend = :bottomright,
    c = :orange,
)

p = plot!(
    [minimum(neuron_average), maximum(neuron_average)],
    [minimum(neuron_average), maximum(neuron_average)],
    title = "Population activity, per sample",
    lc = :grey,
    ls = :dash,
    label = "",
)
plot!(
    p1,
    legend_backgroundcolor = :transparent,
    xlabel = "Input CBA (Hz)",
    ylabel = "Input ABC (Hz)",
)
p_rate = plot(p1, p2, size = (900, 400), margin = 5Plots.mm)
path =
    plotsdir("sequence_detection", "recurrent_connection", "comparison_activity.pdf") |>
    x -> (mkpath(dirname(x)); x)
savefig(p_rate, path)

import HypothesisTests: EqualVarianceTTest, pvalue
ttest_result = EqualVarianceTTest(correct, false_positive)
@info "t-test: $(ttest_result)"
p_rate

##
# ## Perform a one-sample t-test on the difference between two observations
# ## Here, `a` is assumed to be a 2D array where the values along dimension 1 are to be compared

## Print the p-value inf

# %%


##
## Import OneSampleTTest from HypothesisTests module
# W_end = [mean(syn.W) for syn in inputs_syn] # It seems this line is not used as per your instruction

## Create violin plot to visualize the distribution and annotate with the p-value
using SparseArrays

##


# %%
most_active
SNN.clear_monitor(network.pop.E)
# SNN.monitor(network.pop.E, [:v_s, :v_d1])
# SNN.monitor(network.pop.E, [:fire])
SNN.monitor(network.pop.E, [(:v_s, most_active), (:v_d1, most_active)])
SNN.monitor(network.pop.E, [(:fire, most_active)])
# %%
SNN.clear_records([network.pop...])
@unpack E = network.pop
# back_syn.exc_s.W .=1
# back_pop.Es.rate .= 2500Hz
# back_pop.Ed.rate .= 200Hz
train!(populations, synapses, duration = 2000ms)
SNN.raster([network.pop...])

SNN.vecplot!(plot(), network.pop.E, :v_d1, r = 100ms:200ms; neurons = most_active)
##
# Simulate for 500ms without input
# SNN.monitor(network.pop.E, [(:v_s, most_active), (:v_d1, most_active), (:fire, most_active)])
input_rate = 1000Hz
SNN.clear_records(network.pop.E)
# Setting input rates to zero
set_input_rate!(inputs.pop, 0, 0Hz)
# Simulating again for 250ms
sim!(populations, synapses, duration = 250ms)
SNN.clear_records(network.pop.E, :fire)
for x in [1, 2, 3]
    set_input_rate!(inputs.pop, x, input_rate)
    sim!(populations, synapses, duration = 50ms)
end
# Simulating for 150ms post input
set_input_rate!(inputs.pop, 0, 0Hz)
sim!(populations, synapses, duration = 150ms)
@info "rate: $(sum(sum(E.records[:fire])))"
# Recording sum of firings
dm = mean(hcat(E.records[:v_d1]...), dims = 1)[1, 1:10:end]
ds = std(hcat(E.records[:v_d1]...), dims = 1)[1, 1:10:end]
sm = mean(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]
ss = std(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]
sm1 = mean(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]
p1 = plot(dm, ribbon = ds, label = "ABC")
p2 = plot(sm, ribbon = ss, label = "ABC")
p1_r = SNN.raster([E], (0, 300ms))

# Other sequence
# Resetting input rates to zero
set_input_rate!(inputs.pop, 0, 0Hz)
sim!(populations, synapses, duration = 1500ms)
# Clearing records after each simulation
SNN.clear_records(network.pop.E)
# Setting input rates to zero
set_input_rate!(inputs.pop, 0, 0Hz)
# Simulating again for 250ms
sim!(populations, synapses, duration = 250ms)
SNN.clear_records(network.pop.E, :fire)
for x in [1, 2, 1]
    set_input_rate!(inputs.pop, x, input_rate)
    sim!(populations, synapses, duration = 50ms)
end
# Simulating for 150ms post input
set_input_rate!(inputs.pop, 0, 0Hz)
sim!(populations, synapses, duration = 150ms)
@info "rate: $(sum(sum(E.records[:fire])))"

# Recording sum of firings

dm = mean(hcat(E.records[:v_d1]...), dims = 1)[1, 1:10:end]
ds = std(hcat(E.records[:v_d1]...), dims = 1)[1, 1:10:end]
sm = mean(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]
sm2 = mean(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]
ss = std(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]
p1 = plot!(p1, dm, ribbon = ds, label = "BCA")
p2 = plot!(p2, sm, ribbon = ss, label = "CBA")
plot!(ylabel = "Membrane potential (mV)", xlabel = "Time (ms)")

p2_r = SNN.raster([E], (0, 300ms), title = "$(sum(E.records[:fire]))")
plot(p1, p2, layout = (1, 2), ylims = (-70, 10))
##
using RollingFunctions
p3 = plot(runmean(sm1 - sm2, 100))


##
SNN.clear_monitor(network.pop.E)
SNN.monitor(network.pop.E, [(:v_s, most_active), (:v_d1, most_active)])
SNN.monitor(network.pop.E, [(:fire, most_active)])
# %%
sm_diff = map(1:30) do x
    @unpack E = network.pop
    back_pop.E.rate .= 50Hz
    input_rate = 100Hz
    SNN.clear_records(network.pop.E)
    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 250ms)
    SNN.clear_records(network.pop.E, :fire)
    for x in [1, 2, 3]
        set_input_rate!(inputs.pop, x, input_rate)
        sim!(populations, synapses, duration = 50ms)
    end
    # Simulating for 150ms post input
    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 150ms)
    @info "rate: $(sum(sum(E.records[:fire])))"
    sm1 = mean(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]

    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 1500ms)
    SNN.clear_records(network.pop.E)
    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 250ms)
    SNN.clear_records(network.pop.E, :fire)
    for x in [3, 2, 1]
        set_input_rate!(inputs.pop, x, input_rate)
        sim!(populations, synapses, duration = 50ms)
    end
    set_input_rate!(inputs.pop, 0, 0Hz)
    sim!(populations, synapses, duration = 150ms)
    @info "rate: $(sum(sum(E.records[:fire])))"
    sm2 = mean(hcat(E.records[:v_s]...), dims = 1)[1, 1:10:end]
    sm1 - sm2
end

##
using RollingFunctions
mu = runmean(mean(sm_diff), 20)
si = runmean(std(sm_diff), 20)
plot(mu, ribbon = si)
hline!([0], lc = :black)


histogram(network.syn.E_to_E.W)
# %%
plot(p1_r, p2_r, layout = (2, 1))

# %%
@unpack E_to_E = network.syn
@unpack W = E_to_E

# %%
histogram(W)

# %%

# %%
