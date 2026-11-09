# Common set-up of the tutorials that save figures: activates the project, loads the packages the examples
# use (SpikingNeuralNetworks, SNNUtils, SNNPlots with the Makie defaults, CairoMakie, Distributions, ...) and defines
# `ASSET_PATH`, the folder where figures are written (figures/ at the root of the repository).
using DrWatson
findproject() |> quickactivate

using Distributions
using UnPack
using Random
using Statistics
using StatsBase
using LaTeXStrings
using CairoMakie

using SpikingNeuralNetworks
import SNNPlots: vecplot
using SNNPlots
using SNNUtils

SNN.@load_units
SNNPlots.@makie_default

ASSET_PATH = joinpath(@__DIR__, "..", "figures")
mkpath(ASSET_PATH)
