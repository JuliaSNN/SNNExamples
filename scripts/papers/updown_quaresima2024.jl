using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

## Define the network parameters

function run_model(;exc_params)
    # Number of neurons in the network
    NE = 2500
    NI = NE ÷ 4
    NI1 = round(Int,NI * 0.35)
    NI2 = round(Int,NI * 0.65)

    # Import models parameters
    I1_params = duarte2019.PV
    I2_params = duarte2019.SST
    @unpack connectivity, plasticity = quaresima2023
    @unpack dends, NMDA, param, soma_syn, dend_syn = exc_params

    # Define interneurons I1 and I2
    I1 = SNN.IF(; N = NI1, param = I1_params, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = I2_params, name="I2_sst")
    E = SNN.BallAndStickHet(N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="Exc")
    # background noise
    noise = Dict(
        # :noise_s   => SNN.PoissonStimulus(E,  :he_s,  param=1.0kHz, cells=:ALL, μ=0.f0, name="noise_s",),
        :d   => SNN.PoissonStimulus(E,  :he_d,  param=2.0kHz, cells=:ALL, μ=5.f0, name="noise_s",),
        :i1  => SNN.PoissonStimulus(I1, :ge,   param=1.5kHz, cells=:ALL, μ=1.f0,  name="noise_i1"),
        :i2  => SNN.PoissonStimulus(I2, :ge,   param=2kHz, cells=:ALL, μ=1.8f0, name="noise_i2")
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
    #
    network = merge_models(pop, noise=noise, syn)
    @unpack W = network.syn.E_to_E
    # W[rand( 1:length(W), 400)] .= 30.0

    SNN.train!(model=network, duration= 5s, pbar=true, dt=0.125)
    SNN.monitor([network.pop...], [:fire, :v_d, :v_s, :v, (:g_d, [10,20,30,40,50]), (:ge_s, [10,20,30,40,50]), (:gi_s, [10,20,30,40,50])], sr=200Hz)
    mytime = SNN.Time()
    SNN.train!(model=network, duration= 10s, pbar=true, dt=0.125, time=mytime)
    return network
end

model_nmda = run_model(exc_params=quaresima2022)
plot_activity(model_nmda, 8s:2ms:10s)
vecplot(model_nmda.pop.E, :v_d, neurons =1, r=8s:10s,label="soma")
vecplot(model_nmda.pop.E, :v_s, neurons =199, r=7s:0.4:10s,label="soma")
##
model_nonmda = run_model(exc_params=quaresima2022_nar(0.2))
 
histogram(model_nmda.syn.I2_to_E.W)

# %%
# Run the model
# %%
## Target activation with stimuli

# Trange = 1s:10ms:get_time(mytime)

##
plot_activity(model_nonmda, 4s:2ms:10s)
histogram(getvariable(model_nonmda.pop.E, :v_s)[:])
vecplot(network.pop.E, :v_s, neurons =1, r=0s:10s,label="soma")

histogram(getvariable(network.pop.E, :v_s)[:])
histogram(getvariable(network.pop.E, :v_s)[:])
