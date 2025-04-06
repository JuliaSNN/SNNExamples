using DrWatson
quickactivate("../../")
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Distributions
using Random
using Statistics
using YAML
using PyCall
using DataFrames
using DataFramesMeta
using CSV

# Load configuration
root = YAML.load_file(projectdir("conf.yml"))["paths"]["local"]
data_path = joinpath(root, "working_memory", "Seeholzer") |> mkpath
include("model.jl")
##

function run_model(conn)
    # Define IFParameterSingleExponential structs for E and I neurons using the parameters from Ep and Ip
    E_param = SNN.IFParameterSingleExponential(
        τm = 20ms,
        Vt = -50mV,
        Vr = -60mV,
        El = -70mV,
        R = 1/25nS,
        τe = 100ms,  # Single time constant for excitatory synapses
        τi = 10ms,  # Single time constant for inhibitory synapses
        E_i = -70mV,
        E_e = 0mV,
        τabs = 2ms,
    )
    I_param = SNN.IFParameterSingleExponential(
        τm = 10ms,
        Vt = -50mV,
        Vr = -60mV,
        El = -70mV,
        R = 1/ 20nS,
        τe = 100ms,  # Single time constant for excitatory synapses
        τi = 10ms,  # Single time constant for inhibitory synapses
        E_i = -70mV,
        E_e = 0mV,
        τabs = 1ms,
    )
    # Example usage of the IFParameter structs

    E = IF(N=1600, param=E_param)
    I = IF(N=400, param=I_param)

    # WI = linear_network(E.N, σ_w=0.38, w_max=1.4)[1:I.N, 1:E.N]
    # E_to_I = SNN.SpikingSynapse(E,I, :ge; w=WI)
    E_to_I = SNN.SpikingSynapse(E,I, :ge; μ=conn.E_to_I, p=0.2, σ=0)
    I_to_I = SNN.SpikingSynapse(I,I, :gi; μ=conn.I_to_I, p=0.2, σ=0)
    I_to_E = SNN.SpikingSynapse(I,E, :gi; μ=conn.I_to_E, p=0.2, σ=0)
    W = linear_network(E.N, σ_w=conn.σ, w_max=conn.w_max)
    E_to_E = SNN.SpikingSynapse(E,E, :ge; w=W, param=STPParameter(
        τD= 150ms, # τx
        τF= 650ms, # τu
        U = 0.2f0,
    ))

    ExcNoise = CurrentStimulus(E;
        param = CurrentNoiseParameter(E.N; I_base = 200pF, I_dist=Normal(250pF,450pF), α=0.5f0)
    )

    InhNoise = CurrentStimulus(I;
        param = CurrentNoiseParameter(I.N; I_base =100pF, I_dist=Normal(2pF,1pF), α=1f0)
    )

    model = merge_models(;
        E,
        I,
        E_to_I,
        I_to_I,
        I_to_E,
        E_to_E,
        ExcNoise,
        InhNoise,
        silent = true
    )

    input_neurons = 800:900
    input_rates = zeros(1:model.pop.E.N)
    # param = PoissonStimulusInterval(rate=input_rates, intervals=[[5s,6s]] )
    # stim = PoissonStimulus(model.pop.E, :ge; param)

    # model = merge_models(;model..., ext_input=stim)

    SNN.monitor(model.pop, [:fire])
    SNN.monitor(model.syn.E_to_E, [:u, :x], sr=200Hz)
    SNN.monitor(model.syn.E_to_E, [:ρ], sr=40Hz)
    train!(;model, duration=5s, dt=0.125ms, pbar=true)
    model.stim.ExcNoise.param.I_base[input_neurons] .+= 400pF
    train!(;model, duration=1s, dt=0.125ms, pbar=true)
    model.stim.ExcNoise.param.I_base[input_neurons] .-= 400pF
    train!(;model, duration=5s, dt=0.125ms, pbar=true)

    return model
end


conn = (
    E_to_I = 1.29,
    I_to_I = 2.7,
    I_to_E = 1.8,
    σ = 0.38,
    w_max = 0.13,
)
model = run_model(conn)
raster(model.pop, 4s:10s, every=5)

record(model.syn.E_to_E, :ρ)

p = vecplot(model.syn.E_to_E, :ρ, interval = 4s:10s, dt=0.125ms, neurons=800:900, pop_average=true, title="E Neuron Voltage")
vecplot!(p,model.syn.E_to_E, :x, interval = 4s:10s, dt=0.125ms, neurons=200:300, pop_average=true, title="E Neuron Voltage")
plot!(ylims=:auto)

p = vecplot(model.syn.E_to_E, :u, interval = 4s:10s, dt=0.125ms, neurons=800:900, pop_average=true, title="E Neuron Voltage")
vecplot!(p,model.syn.E_to_E, :u, interval = 4s:10s, dt=0.125ms, neurons=200:300, pop_average=true, title="E Neuron Voltage")
plot!(ylims=:auto)


p = vecplot(model.syn.E_to_E, :x, interval = 4s:10s, dt=0.125ms, neurons=800:900, pop_average=true, title="E Neuron Voltage")
vecplot!(p,model.syn.E_to_E, :x, interval = 4s:10s, dt=0.125ms, neurons=200:300, pop_average=true, title="E Neuron Voltage")
plot!(ylims=:auto)

model_loss(model, 20s:30s)

# cv[isnan.(cv)] .= 0.0  # Replace NaN values with 0.0
# plot(cv, title="CV", xlabel="Time (s)", ylabel="CV", label="CV")
# ##

fr, r, labels = firing_rate(model.pop, interval=20s:30s, pop_average=true)
plot(r, fr, title="Firing Rate", xlabel="Time (s)", ylabel="Firing Rate (Hz)", label=labels)
# ##
# #
# #
# vecplot(model.pop.E, :v, interval = 7s:10s, dt=0.125ms, neurons=1:1, title="E Neuron Voltage")
# ##

# using SpecialFunctions


# heatmap(W)


##
begin
    study_name = joinpath(data_path, "attractor_size")
    storage_name = "sqlite:///$(study_name).db"
    optuna = pyimport("optuna")
    ## Optuna
end
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=true)

df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
csv_path = joinpath(data_path, "optuna_WM_parameters.csv") 
df.to_csv(csv_path)
df = DataFrame(CSV.File(csv_path))

config = study.trials[291].params |> x-> Dict{String, Float32}(x) |> dict2ntuple
model = run_model(config)
raster(model.pop, 20s:30s, every=1)
model_loss(model, 20s:10ms:30s)


##
att_df = @rsubset df :values_0<20
att_df = @rsubset! df :values_1<200

scatter(att_df.params_I_to_E, df.params_E_to_I, att_df.values_1.+20, camera=(60,30), 
    xlabel="I to E", ylabel="E to I", zlabel="Inhibitory Firing rate", title="Attractor Width")
scatter(att_df.params_I_to_E, df.params_E_to_I, att_df.values_2, camera=(60,30), 
    xlabel="I to E", ylabel="E to I", zlabel="Inhibitory Firing rate", title="Attractor Width")

