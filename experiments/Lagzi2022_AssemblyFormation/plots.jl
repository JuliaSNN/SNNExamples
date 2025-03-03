using Plots
using SpikingNeuralNetworks

function learning_plot(model)
    @info "Loading model with τ=$(τs[t]) and rate=$(stim_rates[r])"
    stim_rate = stim_rates[r]
    stim_τ = τs[t]
    info = (τ= stim_τ, rate=stim_rate)
    plot_path = plotsdir("Lagzi2022_AssemblyFormation", savename(info)) |> mkpath
    model = load_data(path, "Model_sst", info).model

    # %%
    W1,r = record(model.syn.E1_to_E1, :W)
    W2,r = record(model.syn.E1_to_E2, :W)
    r = 0:2s:r[end]
    p1 = mean(W1[:,r], dims=1)[1,:] |> x->plot(r, x, label="E1 to E1", lw=4)
    p1 = mean(W2[:,r], dims=1)[1,:] |> x->plot!(r, x, label="E1 to E2", lw=4)
    plot!(p1, xlabel="Time (ms)", ylabel="Weight", title="E1 to E1 and E1 to E2", size=(800, 300), margin=5Plots.mm)
    p2 = histogram(model.syn.E1_to_E1.W, bins=0.1:0.01:2.0, c=:darkblue, alpha=0.5, label="E1 to E1")
    histogram!(model.syn.E1_to_E2.W, bins=0.1:0.01:2.0, c=:darkred, alpha=0.5, label="E1 to E2")
    fig = plot(p1, p2, layout=(2,1), size=(800, 600), margin=5Plots.mm)
    savefig(fig, joinpath(plot_path, "exc_weight.svg"))
    fig
    ##

    # %%
    lags, corr11 = compute_covariance_density(merge_spiketimes(spiketimes(model.pop.E1)),merge_spiketimes(spiketimes(model.pop.E1)))
    lags, corr12 = compute_covariance_density(merge_spiketimes(spiketimes(model.pop.E1)),merge_spiketimes(spiketimes(model.pop.E2)))
    plot(lags, corr11, xlabel="Time lag (ms)", ylabel="Correlation", title="Cross-correlogram", size=(800, 300), margin=5Plots.mm)
    plot!(lags, corr12, xlabel="Time lag (ms)", ylabel="Correlation", title="Cross-correlogram", size=(800, 300), margin=5Plots.mm)

    # %%
    fig = raster(model.pop, 195s:205s, every=5, size=(800, 500), margin=5Plots.mm, link=:x, legend=:none,yrotation=0)
    savefig(fig,joinpath(plot_path, "raster_long.svg"))
    fig = raster(model.pop, 202:205s, every=5, size=(800, 500), margin=5Plots.mm, link=:x, legend=:none,yrotation=0)
    savefig(fig,joinpath(plot_path, "raster_short.svg"))
    ##

    plot()
    for (n, k) in enumerate(keys(model.syn))
        syn = model.syn[k]
        syn.param isa SNN.no_STDPParameter && continue 
        μ0 = n>=8 ? μ*JInh : μ
        W = median(syn.W)/μ0
        bar!([string(k)], [W], size=(800, 300), margin=5Plots.mm, link=:x, legend=:none)
    end
    fig = plot!(xlabel="Synapse", ylabel="Weight", title="Synaptic weights", xrotation=45, bottommargin=10Plots.mm, size=(800, 300), margin=5Plots.mm)
    savefig(fig, joinpath(plot_path, "synaptic_weights.svg"))
    fig
    ##

    # %% [markdown]
    # Show the correlations between the firing rates of the two populations


    # %%
    frs, r, names = firing_rate(model.pop, interval = 0s:50ms:500s, interpolate=false, mean_pop=true, τ=50ms)
    plot(r./1000,frs[1:2], xlims=(300,400))
    ##
    plots = []
    labels = ["E1", "E2", "PV", "SST1", "SST2"]
    for n in 1:5
        for m in 1:5
            labx = n == 5 ? labels[m] : ""
            laby = m == 1 ? labels[n] : ""
            z = plot()
            try 
                z = histogram2d(frs[n], frs[m],xlabel=labx, ylabel=laby, legend=false, xlims=(0,100), ylims=(0,100), c=:amp)
            catch e
            end
            push!(plots, z)

        end
    end
    fig = plot(plots..., layout=(5,5), size=(800, 800), margin=0Plots.mm)
    savefig(fig, joinpath(plot_path, "population_firing_correlation.svg"))
    ##
        # cor(frs[1][1:4_000], frs[5][1:4_000])
        # cor(frs[1][end-4_000:end], frs[2][end-4_000:end])

    p = histogram(model.syn.E1_to_E1.W,   bins=0.1:0.01:1.3,c=:darkblue, alpha=0.5, label="E1 to E1")
    histogram!(model.syn.E1_to_E2.W, bins=0.1:0.01:1.3, c=:darkred, alpha=0.5, label="E1 to E2")

    ##
    q = histogram(model.syn.SST1_to_E1.W, c=:darkblue, alpha=0.5, label="SST1 to E1", bins=0:0.5:50)
    histogram!(model.syn.SST1_to_E2.W, c=:darkred, alpha=0.5, label="SST1 to E2", bins=0:0.5:50)
    ##

    ##
    plots = []
    labels = ["E1", "E2", "PV", "SST1", "SST2"]
    for n in 1:5
        for m in 1:5
            labx = n == 5 ? labels[m] : ""
            laby = m == 1 ? labels[n] : ""
            t_end = 40
            rr = vcat(-reverse(r[1:t_end].-r[1]),0, r[1:t_end].-r[1])
            cc1 = crosscov(frs[m], frs[n], -t_end:1:t_end) 
            ylims = maximum(abs.(cc1))
            pp = plot(rr, cc1, label="Tuned", legendfontsize=13, lw=4, xlabel=labx, ylabel=laby, ylims=(-ylims,ylims), frame=:origin, ticks=:none)
            push!(plots, pp)
        end
    end
    fig = plot(plots..., layout=(5,5), size=(800, 800), margin=5Plots.mm, legend=false)
    savefig(fig, joinpath(plot_path, "population_firing_crosscorrelation.svg"))
    fig
    ##
end

function response_plot(recs, models)
    EEs = [mean(model.syn.SST1_to_E1.W)*model.pop.SST1.N for model in models]
    color = palette(:roma, length(recs))
    responses = []
    p2 = plot()
    p3 = plot()
    for n in eachindex(recs)
            @unpack recordings, info, rec_interval = recs[n]
            fr = mean(recordings[:,:,1], dims=2)[:,1]
            fr0 = fr
            fr = fr .- mean(fr[1:1000])
            plot!(p2,rec_interval,fr, label=NSSTs[n], c=color[n], )
            plot!(p3,rec_interval,fr0, label=NSSTs[n], c=color[n], )
            push!(responses,mean(fr[1500:1800,:,1]))

    end
    p1 = scatter(EEs, responses, label="E1", xlabel="SST1_to_E1", ylabel="E1 response", legend=:topleft)
    plot!(p2, xlims=(0.3s, 1.9s), legendtitle="NSST (%)", size=(800, 800), xlabel="Time (s)", ylabel="ΔFiring rate (Hz)", margin=5Plots.mm, )
    plot!(p3, legendtitle="NSST (%)", size=(800, 800), xlabel="Time (s)", ylabel="Firing rate (Hz)", margin=5Plots.mm)
    vline!(p2,[1s, 1.5s], label="", ls=:dash, lc=:black, ylims=(-4, 20))
    vline!(p3,[1s, 1.5s], label="", ls=:dash, lc=:black, ylims=(-4, 35))
    plot(p1,p2,p3, layout=(3,1), size=(800, 800), margin=5Plots.mm, bottommargin=10Plots.mm)
end

