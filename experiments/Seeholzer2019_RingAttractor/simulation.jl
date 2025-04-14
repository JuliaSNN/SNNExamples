using DrWatson
quickactivate(findproject())
include(projectdir("loaders", "analysis.jl"))

# Load configuration
data_path = datadir("working_memory", "Seeholzer") |> mkpath
include("model.jl")

file_path = joinpath(@__DIR__,"network_parameters.csv")
entry = 5 # Example entry
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
        ),
        NE = 800,
        ΔT = 1s,
        input_neurons = [400:500],
        sparsity = .2,
    )#

base_conf = config
config = get_configuration(base_conf, entry, file_path)
config |> dump
model, pre, post = run_task(config)

raster(model.pop, 0s:10ms:15s, every=1)
# objective_values = (width_pre-width_post, abs(cv_pre-1), abs(ff_pre-1))

##
width_pre, fE_pre, fI_pre, cv_pre, ff_pre = model_loss(model, 0s:10ms:5s)
width_post, fE_post, fI_post, cv_post, ff_post = model_loss(model, (ΔT+ 5s):10ms:(10s + ΔT))

width, fE, fI, cv, ff = model_loss(model, 6s:10ms:11s)
##


st = merge_spiketimes(spiketimes(model.pop.E))
bins,_ = SNN.bin_spiketimes(st; time_range = 0:10ms:5s, do_sparse = false)
ff = var(bins) / mean(bins)  # Fano Factor

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

