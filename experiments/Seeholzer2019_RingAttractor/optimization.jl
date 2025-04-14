using Distributed
addprocs(maximum([8-nprocs(), 0]))

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
    root = YAML.load_file(projectdir("paths.yaml"))["kilo_local"]
    data_path = joinpath(root, "working_memory", "Seeholzer") |> mkpath
    include("model.jl")
    study_name = "async_sparsity_facilitation"
    storage_name = joinpath(data_path, "attractors_run4")
    storage_name = "sqlite:///$(storage_name).db"

    function objective(trial)
        # E_to_I = trial.suggest_float("E_to_I", 0.5, 3)
        # I_to_I = trial.suggest_float("I_to_I", 0, 3)
        I_to_E = trial.suggest_float("I_to_E", 0, 3)
        σ = trial.suggest_float("σ", 0, 1)
        w_max = trial.suggest_float("w_max", 0, 1)
        U = trial.suggest_float("U", 0, 1)
        sparsity = trial.suggest_float("sparsity", 0, 1)

        STPparam = STPParameter(
            τD= 150ms, # τx
            τF= 650ms, # τu
            U = U,
        )

        config = (
            E_to_I = 1.29,
            I_to_I = 2.7,
            I_to_E = I_to_E,
            σ_w = σ,
            w_max = w_max,
            STPparam = STPparam,
            sparsity = sparsity,
            ΔT = 1s,
            input_neurons = [400:500],
            NE = 800,
        )
        result = run_task(config)
        if isnothing(result)
            return (-10,1,1)
    
        else
            model, pre, post = result
            width_pre, fE_pre, fI_pre, cv_pre, ff_pre = model_loss(model, pre)
            width_post, fE_post, fI_post, cv_post, ff_post = model_loss(model, post)
            objective_values = (width_pre-width_post, abs(cv_pre-1), abs(ff_pre-1))
    
            trial.set_user_attr("fE_pre", mean(fE_pre))
            trial.set_user_attr("fI_pre", mean(fI_pre))
            trial.set_user_attr("fE_post", mean(fE_post))
            trial.set_user_attr("fI_post", mean(fI_post))
            trial.set_user_attr("width_pre", width_pre)
            trial.set_user_attr("width_post", width_post)
            trial.set_user_attr("cv_pre", cv_pre)
            trial.set_user_attr("cv_post", cv_post)
    
            return objective_values
        end
    end
    ##
end

objective_targets = ["maximize", "minimize", "minimize"]
objective_names = ["width", "cv", "ff"]
study = optuna.create_study(directions=objective_targets,  study_name=study_name, storage=storage_name, load_if_exists=true)
study.set_metric_names(objective_names)

@everywhere begin
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=true)
end 

# Create a study and optimize the objective function
tasks = @sync @distributed for w in workers()
    sleep(rand(1:140))
    @info "Worker: $(getpid()) active"
    study.optimize(objective, n_trials=40)
end
wait(tasks)
