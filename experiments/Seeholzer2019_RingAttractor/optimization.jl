using Distributed
addprocs(maximum([15-nprocs(), 0]))

@everywhere using DrWatson
@everywhere quickactivate("../../")
@everywhere using Revise
@everywhere using SpikingNeuralNetworks
@everywhere SpikingNeuralNetworks.@load_units;
@everywhere using SNNUtils
@everywhere using Plots
@everywhere using Distributions
@everywhere using Random
@everywhere using Statistics
@everywhere using YAML
@everywhere using PyCall
@everywhere optuna = pyimport("optuna")


# Load configuration

@everywhere begin
    root = YAML.load_file(projectdir("conf.yml"))["paths"]["local"]
    data_path = joinpath(root, "working_memory", "Seeholzer") |> mkpath
    include("model.jl")
    study_name = joinpath(data_path, "attractor_size")
    storage_name = "sqlite:///$(study_name).db"

    function objective(trial)
        E_to_I = trial.suggest_float("E_to_I", 0, 3)
        I_to_I = trial.suggest_float("I_to_I", 0, 3)
        I_to_E = trial.suggest_float("I_to_E", 0, 3)
        σ = trial.suggest_float("σ", 0, 1)
        w_max = trial.suggest_float("w_max", 0, 1)
        conn = (
            E_to_I = E_to_I,
            I_to_I = I_to_I,
            I_to_E = I_to_E,
            σ = σ,
            w_max = w_max,
        )
        model = run_model(conn)
        return model_loss(model, 20s:10ms:30s)
    end
end

@everywhere study = optuna.create_study(directions=["minimize", "minimize", "minimize"], study_name=study_name, storage=storage_name, load_if_exists=true)
# study = optuna.create_study(study_name=study_name, storage=storage_name)


# Create a study and optimize the objective function
tasks = []
for w in workers()
    p = @spawnat w begin
        @info "Worker: $(getpid()) active"
        study.optimize(objective, n_trials=20)
    end
    push!(tasks, p)
end
[fetch(t) for t in tasks]
