# Graph of a model: `SNN.graph(model)` returns a MetaGraph whose vertices are the populations and stimuli and
# whose edges are the synapses and stimuli connections. Drawn here with a circular layout in CairoMakie (the
# original used GraphMakie with an interactive GLMakie window).
# Runtime: about 40 s.
using SpikingNeuralNetworks
SNN.@load_units
using Graphs
using MetaGraphs
using CairoMakie

# Define the network
network = let
    N = 1000
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV), name = "E")
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "I")
    E_to_I = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 3.0))
    E_to_E = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5))
    I_to_I = SNN.SpikingSynapse(I, I, :gi; conn = (p = 0.2, μ = 4.0))
    I_to_E = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 1), LTPParam = SNN.iSTDPRate(r = 4Hz))
    norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 30ms), [E_to_E])
    (pop = (; E, I), syn = (; I_to_E, E_to_I, E_to_E, norm, I_to_I))
end

# Create background for the network simulation
noise = SNN.PoissonStimulus(network.pop.E, :ge; param = SNN.PoissonFixed(rate = 2.8kHz), neurons = :ALL)
noise2 = SNN.PoissonStimulus(network.pop.E, :gi; param = SNN.PoissonFixed(rate = 3kHz), neurons = :ALL)
model = SNN.compose(network, noise = noise, noise2 = noise2, silent = true)

my_graph = SNN.graph(model)
@info "Graph with $(nv(my_graph)) vertices and $(ne(my_graph)) edges"

## Circular layout
n = nv(my_graph)
pos = [Point2f(cos(2π * (k - 1) / n), sin(2π * (k - 1) / n)) for k = 1:n]
names = [get_prop(my_graph, v, :name) for v in vertices(my_graph)]
fig = Figure(size = (600, 600))
ax = Axis(fig[1, 1], aspect = DataAspect())
hidedecorations!(ax); hidespines!(ax)
for e in edges(my_graph)
    a, b = pos[src(e)], pos[dst(e)]
    if src(e) == dst(e)       # recurrent connection: small loop outside the node
        c = 1.25f0 * a
        lines!(ax, [Point2f(c .+ 0.12f0 .* (cos(t), sin(t))) for t in range(0, 2π, 30)], color = :gray)
    else
        lines!(ax, [a, b], color = :gray, linewidth = 1)
        arrows2d!(ax, [a + 0.85f0 * (b - a)], [0.08f0 * (b - a)], color = :gray)
    end
end
scatter!(ax, pos, markersize = 30, color = :steelblue)
text!(ax, pos, text = string.(names), align = (:center, :bottom), offset = (0, 20))
fig
