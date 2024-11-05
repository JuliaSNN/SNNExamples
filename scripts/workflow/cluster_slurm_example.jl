#!/bin/bash

#SBATCH --job-name="test_run"            # Job Name
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1                       # 1 CPU allocation per Task
#SBATCH --mem=1GB
#SBATCH --time=1:00:00
#SBATCH --partition=common
#SBATCH -q common
#SBATCH -e slurm-test_julia_%j.err
#SBATCH -o slurm-test_julia_%j.out
#SBATCH --qos=fast

#=
srun julia $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
exit
# =#


using Distributed

# launch worker processes
addprocs(4)

println("Number of processes: ", nprocs())
println("Number of workers: ", nworkers())

# each worker gets its id, process id and hostname
for i in workers()
    id, pid, host = fetch(@spawnat i (myid(), getpid(), gethostname()))
    println(id, " " , pid, " ", host)
end

# remove the workers
for i in workers()
    rmprocs(i)
end