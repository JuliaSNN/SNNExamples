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


config = (
    E_to_I = 1.29,
    I_to_I = 2.7,
    I_to_E = 1.8,
    σ_w = 0.38,
    w_max = 0.73,
    STPparam = STPParameter(
        τD= 150ms, # τx
        τF= 650ms, # τu
        U = 0.4f0,
    )
)
input_neurons = [400:500, 100:200]
model = run_model(config, input_neurons, 2s)
raster(model.pop, 0s:25s, every=5)

##
ρ, r = record(model.syn.E_to_E, :ρ, interpolate=true)

w_1 = indices(model.syn.E_to_E, input_neurons[1], input_neurons[1])
w_2 = indices(model.syn.E_to_E, input_neurons[2], input_neurons[2])
w_out = indices(model.syn.E_to_E, 1:100, input_neurons[2])
plot(r, mean(ρ[w_1,r] , dims=1)[1,:])
plot!(r, mean(ρ[w_2,r], dims=1)[1,:])
plot!(r, mean(ρ[w_out,r], dims=1)[1,:])
##
p1 = vecplot(model.syn.E_to_E, :u, interval = 4s:10s, dt=0.125ms, neurons=700:900, pop_average=true, title="E Neuron Voltage")
vecplot!(p1,model.syn.E_to_E, :u, interval = 4s:10s, dt=0.125ms, neurons=200:300, pop_average=true, title="E Neuron Voltage")
plot!(ylims=:auto)


p2 = vecplot(model.syn.E_to_E, :x, interval = 4s:10s, dt=0.125ms, neurons=700:900, pop_average=true, title="E Neuron Voltage")
vecplot!(p2,model.syn.E_to_E, :x, interval = 4s:10s, dt=0.125ms, neurons=200:300, pop_average=true, title="E Neuron Voltage")
plot!(ylims=:auto)

plot(p1, p2, layout=(2,1), size=(800, 600))
##

# model_loss(model, 20s:30s)

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

