# Model, protocol and analysis functions of persistent_adapting_activity.jl: assemblies of E/I neurons coupled by
# lateral excitation, with plastic I -> E synapses whose rule is a parameter, and a Poisson cue to one assembly.
using Random, Statistics

# Define each of the network recurrent assemblies
function attractor_model(config)
    function get_subnet(name, config)
        N = config.N
        E = SNN.AdEx(N = N, param = config.adex, spike = SNN.PostSpike(At = 1mV), name = "Exc_$name")
        I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "Inh_$name")
        E_to_I = SNN.SpikingSynapse(E, I, :he; conn = config.E_to_I, name = "E_to_I_$name")
        E_to_E = SNN.SpikingSynapse(E, E, :he; conn = config.E_to_E, name = "E_to_E_$name")
        I_to_I = SNN.SpikingSynapse(I, I, :hi; conn = config.I_to_I, name = "I_to_I_$name")
        I_to_E = SNN.SpikingSynapse(I, E, :hi; conn = config.I_to_E, LTPParam = config.istdp, name = "I_to_E_$name")
        norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 10ms), [E_to_E])
        noise = SNN.PoissonStimulus(E, :he; param = SNN.PoissonFixed(rate = config.noise), neurons = :ALL, name = "noise_$name")
        SNN.monitor!([E, I], [:fire])
        SNN.monitor!(I_to_E, [:W], sr = 10Hz)
        return SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, norm, noise, silent = true)
    end

    n_assemblies = config.n_assemblies
    subnets = NamedTuple(Symbol("sub_$n") => get_subnet(n, config) for n = 1:n_assemblies)
    syns = Dict{Symbol,Any}()
    for i = 1:n_assemblies, j = 1:n_assemblies
        i == j && continue
        pre, post = subnets[i], subnets[j]
        push!(syns, Symbol("$(i)E_to_$(j)I_lateral") => SNN.SpikingSynapse(pre.pop.E, post.pop.I, :he; conn = config.lateral_EI))
        push!(syns, Symbol("$(i)E_to_$(j)E_lateral") => SNN.SpikingSynapse(pre.pop.E, post.pop.E, :he; conn = config.lateral_EE))
    end
    return SNN.compose(; subnets..., NamedTuple(syns)..., silent = true)
end

function add_stimulus(network, population, interval, config)
    rate = SNN.PoissonInterval(rate = config.ext_stim, intervals = [interval])
    name = Symbol("stim_$(population)_$(randstring(2))")
    trigger = SNN.PoissonStimulus(getfield(network.pop, population), :he; param = rate, neurons = :ALL, name = string(name))
    return SNN.compose(network; NamedTuple{(name,)}((trigger,))..., silent = true)
end

function test_istdp(config)
    model = attractor_model(config)
    SNN.train!(model = model, duration = config.warmup, dt = 0.125f0)
    SNN.clear_records!(model)
    for (pop, interval) in config.intervals
        model = add_stimulus(model, pop, interval, config)
    end
    SNN.train!(model = model, duration = config.duration, dt = 0.125f0)
    return model
end

# Summary figure: mean I -> E weights, excitatory and inhibitory rates, raster
function iSTDP_activity(model, istdp, config; interval = 1s:20ms:config.duration)
    fig = Figure(size = (900, 800))
    i_to_e = [s for (k, s) in pairs(model.syn) if occursin("I_to_E", string(k))]
    ax1 = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Synaptic weight", title = "I to E synapse")
    _, r_t = SNN.record(i_to_e[1], :W, range = true)
    for (n, s) in enumerate(i_to_e)
        lines!(ax1, r_t ./ 1000, vec(mean(SNN.getvariable(s, :W), dims = 1)), label = "pop $n", linewidth = 3)
    end
    axislegend(ax1, position = :lt)
    exc = [p for (k, p) in pairs(model.pop) if endswith(string(k), "_E")]
    inh = [p for (k, p) in pairs(model.pop) if endswith(string(k), "_I")]
    ax2 = Axis(fig[1, 2], xlabel = "Time (s)", ylabel = "Rate (Hz)", title = "Excitatory populations")
    ax3 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Rate (Hz)", title = "Inhibitory populations")
    for (axis, pops) in ((ax2, exc), (ax3, inh)), p in pops
        fr, r = SNN.firing_rate(p; interval = 0s:50ms:SNN.get_time(model), τ = 50ms, pop_average = true)
        lines!(axis, r ./ 1000, vec(fr), label = p.name, linewidth = 2)
    end
    for axis in (ax2, ax3), x in config.intervals
        vlines!(axis, x[2][2] / 1000, color = :black, linestyle = :dash)   # end of the cue (the clock restarts after the warm-up)
    end
    axislegend(ax2, position = :lt)
    ax4 = Axis(fig[2, 2], xlabel = "Time (s)", ylabel = "Neuron")
    SNN.raster!(ax4, model.pop, (SNN.get_time(model) - 10s):SNN.get_time(model), every = 5)
    return fig
end
