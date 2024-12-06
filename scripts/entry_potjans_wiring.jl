
using SGtSNEpi, Random
using Revise
using CairoMakie, Colors, LinearAlgebra
using GLMakie
using Graphs

import StatsBase.mean
using Plots
include("genPotjansWiring.jl")

function grab_connectome(scale)
   

    pot_conn = potjans_layer(scale)
    display(pot_conn)
    # The Graph Network analysis can't handle negative weight values so offset every weight to make weights net positive.
    stored_min = abs(minimum(pot_conn))
    for (ind,row) in enumerate(eachrow(pot_conn))
        for (j,colind) in enumerate(row)
            if pot_conn[ind,j] < 0.0
                pot_conn[ind,j] = pot_conn[ind,j]+stored_min+1.0
            end
        end
        @assert mean(pot_conn[ind,:]) >= 0.0
    end
    Plots.heatmap(pot_conn,xlabel="post synaptic",ylabel="pre synaptic")
    savefig("connection_matrix.png")

    pot_conn
end
scale = 0.015
pot_conn = grab_connectome(scale)
dim = 2
Lx = Vector{Int64}(zeros(size(pot_conn[2,:])))
Lx = convert(Vector{Int64},Lx)

y = sgtsnepi(pot_conn)
cmap_out = distinguishable_colors(
    length(Lx),
    [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

display(SGtSNEpi.show_embedding( y, Lx ,A=pot_conn;edge_alpha=0.15,lwd_in=0.15,lwd_out=0.013,cmap=cmap_out)