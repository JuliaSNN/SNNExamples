#!/bin/bash

#SBATCH --job-name="test_run"            # Job Name
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1                       # 1 CPU allocation per Task
#SBATCH --mem=1GB
#SBATCH --time=1:00:00
#SBATCH --partition=common
#SBATCH -q common
#SBATCH --qos=fast
#SBATCH -e /pasteur/appa/homes/aquaresi/spiking/network_models/logs/errors/slurm-test_julia_%j.err
#SBATCH -o /pasteur/appa/homes/aquaresi/spiking/network_models/logs/out/slurm-test_julia_%j.out

#=
srun julia $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
exit
# =#


using Distributed
using JSON
using Logging

@info "Number of processes: ", nprocs()
@info "Number of workers: ", nworkers()

open("julia_env.txt","a") do io
    JSON.print(io, ENV)
end
# launch worker processes
addprocs(4)


# each worker gets its id, process id and hostname
for i in workers()
    id, pid, host = fetch(@spawnat i (myid(), getpid(), gethostname()))
    open("julia_run_$(rand(1:100)).txt","a") do io
        println(io, id, " " , pid, " ", host)
    end
end

# remove the workers
for i in workers()
    rmprocs(i)
end