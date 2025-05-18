
function init_model(config; W=nothing)
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

    E = IF(N=config.NE, param=E_param)
    I = IF(N=config.NE ÷ 4, param=I_param)
    #
    @unpack E_to_I, I_to_I, I_to_E, STPparam, σ_w, w_max, sparsity = config

    E_to_I = SNN.SpikingSynapse(E,I, :ge; μ=E_to_I, p=sparsity, σ=0)
    I_to_I = SNN.SpikingSynapse(I,I, :gi; μ=I_to_I, p=sparsity, σ=0)
    I_to_E = SNN.SpikingSynapse(I,E, :gi; μ=I_to_E, p=sparsity, σ=0)

    W = isnothing(W) ? linear_network(E.N, σ_w=σ_w, w_max=w_max) : W 
    E_to_E = SNN.SpikingSynapse(E,E, :ge; w=W, STPParam=STPparam
    )

    ExcNoise = CurrentStimulus(E;
        param = CurrentNoiseParameter(E.N; I_base = 100pF, I_dist=Normal(350pF,250pF), α=0.5f0)
    )

    InhNoise = CurrentStimulus(I;
        param = CurrentNoiseParameter(I.N; I_base =100pF, I_dist=Normal(100pF,100pF), α=1f0)
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


    SNN.monitor!(model.pop, [:fire])
    train!(;model, duration=5s, dt=0.125ms, pbar=true)
    return model
end

function get_configuration(base_conf, entry::Int, file_path::String)
    df = CSV.read(file_path, DataFrame)
    if entry <= 0 || entry > nrow(df)
        throw(ArgumentError("Entry $entry is out of bounds for the CSV file."))
    end
    row = df[entry, :]
    config = (;base_conf...,
        E_to_I = row.g_EI,
        I_to_I = row.g_II,
        I_to_E = row.g_IE,
        σ_w = row.σ_w,
        w_max = row.g_EE,
        STPparam = MarkramSTPParameter(
            τD = row.STP_τσ * ms, # τx
            τF = row.STP_τu * ms, # τu
            U = row.STP_U,
        ),
    )
    return config
end


function model_loss(model, interval)
    attractor_width, _ = is_attractor_state(model.pop.E, interval; ratio=0.3, σ=100.0f0)
    cv, ff = asynchronous_state(model, interval)
    fr = firing_rate(model.pop; interval, τ=20ms, pop_average=true) |> x-> mean.(x[1])
    return attractor_width, abs(fr[1]-3), abs(fr[2]-20), cv, ff
end

function test_WM!(model, input_neurons; input_duration=1s, measure_duration=5s)
    reset_time!(model.time)
    clear_records!(model)
    tic = get_time(model)
    train!(;model, duration=measure_duration, dt=0.125ms, pbar=true)
    toc = get_time(model)
    pre_interval = tic:10ms:toc
    for i in eachindex(input_neurons)## External input on E neurons
        model.stim.ExcNoise.param.I_base[input_neurons[i]] .+= 50pA
        train!(;model, duration=input_duration, dt=0.125ms, pbar=true)
        model.stim.ExcNoise.param.I_base[input_neurons[i]] .-= 50pA
        train!(;model, duration=5s, dt=0.125ms, pbar=true)
    end
    tic = get_time(model)
    train!(;model, duration=measure_duration, dt=0.125ms, pbar=true)
    toc = get_time(model)
    post_interval = tic:10ms:toc
    return pre_interval, post_interval
end

function run_task(config)
    @unpack sparsity, input_neurons, ΔT = config
    W = linear_network(config.NE, σ_w=config.σ_w, w_max=config.w_max)
    sparsify!(W, config.sparsity)
    model = init_model(config; W)
    fr = mean(firing_rate(model.pop.E; interval=0s:10ms:5s, pop_average=true))
    if fr[1] > 20
        return nothing
    end
    # heatmap(W)
    reset_time!(model.time)
    pre, post = test_WM!(model, input_neurons, input_duration=ΔT )
    return model, pre, post
end


function sparsify!(W::Matrix,  p::Real)
    ws  = sample(1:length(W), round(Int,length(W) * (1-p)))
    W[ws] .= 0.0
    return W
end