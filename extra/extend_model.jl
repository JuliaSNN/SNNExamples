using SpikingNeuralNetworks
import SpikingNeuralNetworks:AbstractPopulationParameter
import SpikingNeuralNetworks:AbstractPopulation
using Random

struct TestParams <: AbstractPopulationParameter end

SNN.@snn_kw struct TestModel <: AbstractPopulation
    id::String = randstring(12)
    name::String = "TestModel"
    param::TestParams = TestParams()
    N::Int32 = 100
    x::Vector{Float32} = zeros(N)
    r::Vector{Float32} = zeros(N)
    g::Vector{Float32} = zeros(N)
    I::Vector{Float32} = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::TestModel, param::TestParams, dt::Float32)
    @unpack N, x, r, g, I = p
    @inbounds for i = 1:N
        x[i] += dt * (-x[i] + g[i] + I[i])
        r[i] = tanh(x[i]) #max(0, x[i])
    end
end

function SNN.integrate!(p::TestModel, dt::Float32)
    integrate!(p, p.param, dt)
end


model = merge_models(;pop1 = TestModel(name="ExtendedModel", N=200), silent=true)
sim!(model)


================
Model: Balanced network
----------------
Populations (2):
E         : IF        :  4000       IFParamete
I         : IF        :  1000       IFParamete
----------------
Synapses (4): 
E_to_E             : E -> E.ge                     :          : NoLTP      : NoSTP     
E_to_I             : E -> I.ge                     :          : NoLTP      : NoSTP     
I_to_E             : I -> E.gi                     :          : NoLTP      : NoSTP     
I_to_I             : I -> I.gi                     :          : NoLTP      : NoSTP     
----------------
Stimuli (2):
noiseE     : noiseE -> E.ge                 PoissonStimulus
noiseI     : noiseI -> I.ge                 PoissonStimulus
================