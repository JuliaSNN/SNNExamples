using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

## Define the network parameters

function run_model(; exc_params)
	# Number of neurons in the network
	NE = 2500
	NI = NE ÷ 4
	NI1 = round(Int, NI * 0.35)
	NI2 = round(Int, NI * 0.65)

	# Import models parameters
	I1_params = duarte2019.PV
	I2_params = duarte2019.SST
	@unpack connectivity, plasticity = quaresima2023
	@unpack dends, NMDA, param, soma_syn, dend_syn = exc_params

	# Define interneurons I1 and I2
	I1 = SNN.IF(; N = NI1, param = I1_params, name = "I1_pv")
	I2 = SNN.IF(; N = NI2, param = I2_params, name = "I2_sst")
	E = SNN.BallAndStickHet(N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name = "Exc")
	# background noise
	noise = Dict(
		# :noise_s   => SNN.PoissonStimulus(E,  :he_s,  param=3.0kHz, cells=:ALL, μ=1.f0, name="noise_s",),
		:d  => SNN.PoissonStimulus(E, :he_d, param = 4.0kHz, cells = :ALL, μ = 5.0f0, name = "noise_s"),
		:i1 => SNN.PoissonStimulus(I1, :ge, param = 1.5kHz, cells = :ALL, μ = 1.0f0, name = "noise_i1"),
		:i2 => SNN.PoissonStimulus(I2, :ge, param = 2kHz, cells = :ALL, μ = 1.8f0, name = "noise_i2"),
	)
	syn = Dict(
		:I1_to_I1 => SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...),
		:I1_to_I2 => SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...),
		:I2_to_I2 => SNN.SpikingSynapse(I2, I2, :gi; connectivity.IsIs...),
		:I2_to_I1 => SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...),
		:I1_to_E  => SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.EIf...),
		:I2_to_E  => SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.EdIs...),
		:E_to_I1  => SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...),
		:E_to_I2  => SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...),
		:E_to_E   => SNN.SpikingSynapse(E, E, :he, :d; connectivity.EdE...),
	)
	pop = dict2ntuple(@strdict I1 I2 E)
	#
	network = merge_models(pop, noise = noise, syn)
	@unpack W = network.syn.E_to_E
	# W[rand( 1:length(W), 400)] .= 30.0

	SNN.train!(model = network, duration = 5s, pbar = true, dt = 0.125)
	SNN.monitor([network.pop...], [:fire, :v_d, :v_s, :v, (:g_d, [10, 20, 30, 40, 50]), (:ge_s, [10, 20, 30, 40, 50]), (:gi_s, [10, 20, 30, 40, 50])], sr = 500Hz)
	mytime = SNN.Time()
	SNN.train!(model = network, duration = 10s, pbar = true, dt = 0.125, time = mytime)
	return network
end

model_nmda = run_model(exc_params = quaresima2022)

##
plot_activity(model_nmda, 8s:2ms:10s)
model = model_nmda
p2 = vecplot(model.pop.E, :v_d, neurons = 1, r = 9s:0.1:10s, label = "soma", fill = -70)
vecplot!(p2, model.pop.E, :v_s, neurons = 2, r = 9s:0.125:10s, label = "soma", fill = -70, c = :black)
# raster(spiketimes(model_nmda.pop.E)[1:10], [8s,10s])
# average_firing_rate(model_nmda.pop.E)
##
model_nonmda = run_model(exc_params = quaresima2022_nar(0.2))
##
plot_activity(model_nonmda, 8s:2ms:10s)
vecplot(model_nonmda.pop.E, :v_d, neurons = 1, r = 8s:0.1:10s, label = "soma")
vecplot(model_nonmda.pop.E, :v_s, neurons = 209:210, r = 7s:0.1:10s, label = "soma")
plot(histogram(getvariable(model_nonmda.pop.E, :v_s)[:], xlabel = "", ylabel = "Frequency", title = "Membrane potential distribution", c = :black),
	histogram(getvariable(model_nonmda.pop.E, :v_d)[:], xlabel = "Membrane potential (mV)", c = :black), layout = (2, 1), xlims = (-90, 0), fill = true, legend = false)
# raster(spiketimes(model_nmda.pop.E)[1:10], [8s,10s])
average_firing_rate(model_nonmda.pop.E)
# %%
# Run the model
# %%
## Target activation with stimuli
model = model_nmda

p1 = plot(hist_nmda, hist_nonmda, layout = (1, 2), size = (500, 400), margin = 5Plots.mm, yticks = :none, yaxis = false, plot_title = "Network All-Point-Histogram")
p2 = plot_activity(model_nmda, 8s:2ms:10s)

plots = map(1:11:50) do i
	i = rand(1:2500, 1)
	p = vecplot(model_nmda.pop.E, :v_d, neurons = i, r = 9s:0.1:10s, label = "soma")
	vecplot!(p, model_nmda.pop.E, :v_s, neurons = i, r = 9s:0.1:10s, label = "soma", c = :black)
	plot!(ylims = (-80, 10), xlabel = "", ylabel = "", xticks = :none)
end
plot!(plots[3], ylabel = "Membrane potential (mV)", margin = 5Plots.mm)
plot!(plots[5], xlabel = "Time (s)", margin = 5Plots.mm)
p4 = plot(plots..., layout = (5, 1), size = (800, 800))
pA = plot(p2, p4, layout = (1, 2), size = (1200, 700))


##

hist_nmda = plot(
	histogram(mean(getvariable(model.pop.E, :v_s), dims = 1)[:], xlabel = "", c = :darkblue, norm = true),
	histogram(mean(getvariable(model.pop.E, :v_d), dims = 1)[:], xlabel = "Membrane potential (mV)", c = :darkblue, norm = true),
	layout = (2, 1),
	xlims = (-90, 0),
	fill = true,
	legend = false,
)
model = model_nonmda
hist_nonmda = plot(histogram(mean(getvariable(model.pop.E, :v_s), dims = 1)[:], xlabel = "", c = :darkred, norm = true),
	histogram(mean(getvariable(model.pop.E, :v_d), dims = 1)[:], xlabel = "Membrane potential (mV)", c = :darkred, norm = true), layout = (2, 1), xlims = (-90, 0), fill = true, legend = false)
p1 = plot(hist_nmda, hist_nonmda, layout = (1, 2), size = (500, 400), margin = 5Plots.mm, yticks = :none, yaxis = false, plot_title = "Network All-Point-Histogram")
annotate!(hist_nmda, subplot = 1, (0.05, 0.5), text("Soma", :black, rotation = 90, :center, 11))
annotate!(hist_nmda, subplot = 2, (0.05, 0.5), text("Dendrite", :black, rotation = 90, :center, 11))
annotate!(hist_nonmda, subplot = 1, (0.05, 0.5), text("Soma", :black, rotation = 90, :center, 11))
annotate!(hist_nonmda, subplot = 2, (0.05, 0.5), text("Dendrite", :black, rotation = 90, :center, 11))
##
model = model_nmda
hist_nmda = plot(histogram(getvariable(model.pop.E, :v_s)[:], xlabel = "", ylabel = "Frequency", c = :darkblue, norm = true),
	histogram(getvariable(model.pop.E, :v_d)[:], xlabel = "Membrane potential (mV)", c = :darkblue, norm = true), layout = (2, 1), xlims = (-90, 0), fill = true, legend = false)
plot(hist_nmda, hist_nonmda, layout = (1, 2), size = (800, 400))
model = model_nonmda
hist_nonmda = plot(histogram(getvariable(model.pop.E, :v_s)[:], xlabel = "", c = :darkred, norm = true),
	histogram(getvariable(model.pop.E, :v_d)[:], xlabel = "Membrane potential (mV)", c = :darkred, norm = true), layout = (2, 1), xlims = (-90, 0), fill = true, legend = false)
p3 = plot(hist_nmda, hist_nonmda, layout = (1, 2), size = (500, 400), margin = 5Plots.mm, yticks = :none, yaxis = false, plot_title = "Neuron All-Point-Histogram")



pB = plot(p1, p3, layout = (1, 2), size = (1200, 700))
plot(pA, pB, layout = (2, 1), size = (1200, 1200), margin = 5Plots.mm)
