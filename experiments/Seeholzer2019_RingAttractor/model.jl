
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
    #
    E_to_I = SNN.SpikingSynapse(E,I, :ge; μ=conn.E_to_I, p=0.2, σ=0)
    I_to_I = SNN.SpikingSynapse(I,I, :gi; μ=conn.I_to_I, p=0.2, σ=0)
    I_to_E = SNN.SpikingSynapse(I,E, :gi; μ=conn.I_to_E, p=0.2, σ=0)
    W = linear_network(E.N, σ_w=conn.σ, w_max=conn.w_max)
    E_to_E = SNN.SpikingSynapse(E,E, :ge; w=W)

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


    SNN.monitor(model.pop, [:fire])
    train!(;model, duration=30s, dt=0.125ms)
    return model
end

function model_loss(model, interval=20s:10ms:30s, fr_λ = 0.1)
    attractor_width, _ = is_attractor_state(model.pop.E, interval; ratio=0.3, σ=100.0f0)
    cv, ff = asynchronous_state(model, interval)
    fr = firing_rate(model.pop; interval, τ=20ms, pop_average=true) |> x-> mean.(x[1])
    return attractor_width, abs(fr[2]-20), abs(fr[1]-3) 
end


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