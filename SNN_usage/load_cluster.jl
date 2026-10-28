#!/bin/bash

# Run a network on several worker processes and save one model per worker. The header is a SLURM job
# (`sbatch load_cluster.jl`); run as a plain script (`julia load_cluster.jl`) it starts 4 local workers.
# Runtime: about 55 s.

#=
PROJECTDIR="$HOME/spiking/network_models"
OUTDIR=${PROJECTDIR}/logs/out
ERRDIR=${PROJECTDIR}/logs/errors
echo $PROJECTDIR
# =#

#SBATCH --job-name="test_run"            # Job Name
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1                       # 1 CPU allocation per Task
#SBATCH --mem=1GB
#SBATCH --time=1:00:00
#SBATCH --partition=common
#SBATCH -q common
#SBATCH -e ${ERRDIR}/slurm-test_julia_%j.err
#SBATCH -o ${OUTDIR}/slurm-test_julia_%j.out
#SBATCH --qos=fast

#=
srun julia $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
exit
# =#

using Distributed
addprocs(4; exeflags = "--project=$(Base.active_project())")
@everywhere using DrWatson
@everywhere using SpikingNeuralNetworks
@everywhere SNN.@load_units

# Define the network
@everywhere function create_network()
    E = SNN.AdEx(N = 1000, param = SNN.AdExParameter(Vr = -60mV), name = "E")
    I = SNN.IF(; N = 250, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "I")
    E_to_I = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 3.0))
    E_to_E = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5))
    I_to_I = SNN.SpikingSynapse(I, I, :gi; conn = (p = 0.2, μ = 4.0))
    I_to_E = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 1), LTPParam = SNN.iSTDPRate(r = 4Hz))
    norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 30ms), [E_to_E])
    noise = SNN.PoissonStimulus(E, :ge; param = SNN.PoissonFixed(rate = 2.8kHz), neurons = :ALL)
    model = SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, norm, noise, silent = true)
    SNN.monitor!(model.pop, [:fire])
    SNN.train!(model = model, duration = 5000ms, dt = 0.125f0)
    path = mkpath(datadir("example_cluster"))
    info = (worker = myid(),)
    SNN.save_model(; model, path, name = "network_with_spikes", info, config = info)
    return SNN.get_time(model)
end

# one network per process; fetch so that errors on the workers are reported
futures = [@spawnat p create_network() for p in workers()]
@info "Simulated time on each worker (ms): $(fetch.(futures))"
