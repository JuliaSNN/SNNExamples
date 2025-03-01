
using SpikingNeuralNetworks
# This file contains the parameters for the Lagzi2022_AssemblyFormation experiment.
path = datadir("Lagzi2022_AssemblyFormation")

# Grid search parameters
stim_rates = [0.1, 0.3, 0.5, 0.7]
τs = [10ms, 20ms, 50ms, 100ms, 200ms, 500ms, 1000ms]
NSSTs = 0:10:100

# Model parameters
JInh  = 10 # inhibitory synaptic weight
μ = 0.25mV # mean synaptic weight
ext_rate = 2kHz # external input rate

# STDP parameters
λ = 0.53 # peak difference in the STDP curve LTP and LTD branches 
η_exc =25e-4ms # exc learning rate 
η_sst = 1e-2 # sst learning rate
η_pv = 1e-2 # pv learning rate

# Network configuration
config = 
(
    NE = 400,
    NI = 200,
    NSST = 60,
    adex_param = AdExParameter(a=0, b=0),
    pv_param = IFParameter(El=-55mV),
    sst_param = IFParameter(El=-55mV),
    stim_τ = 100ms,
    stim_rate = 0.5,
    J = JInh,
    EI = (p = 0.4, μ = μ),
    EE = (p = 0.1, μ = μ),
    II = (p = 0.4, μ = JInh*μ),
    IE = (p = 0.4, μ = JInh*μ),
    E_noise = 0.7*ext_rate,
    stdp_exc = STDPParameter(
        A_pre = η_exc,
        A_post =  -λ* η_exc,
        τpost = 30ms,
        τpre = 15ms,
    ) ,
    stdp_sst = STDPParameter(
        A_pre = η_sst,
        A_post =  -λ* η_sst,
        τpost = 30ms,
        τpre = 30ms,
    ) ,
    stdp_pv = STDPParameter(
        A_pre = η_pv,
        A_post =  η_pv,
        τpost = 20ms,
        τpre = 20ms,
    )
)

# @unpack stdp_exc, stdp_sst, stdp_pv = config
# plot(
#     stdp_kernel(stdp_exc, ΔTs= -100:5:100ms, fill=false, title="Exc"),
#     stdp_kernel(stdp_sst, ΔTs= -100:5:100ms, fill=false, title="SST"),
#     stdp_kernel(stdp_pv, ΔTs= -100:5:100ms, fill=false, title="PV"),
#     layout = (1,3),
#     size=(800,300),
#     margin=5Plots.mm,
#     link=:x,
#     ylims=(-4e-2, 4e-2),
#     )
# plot!(
# ylims=(-8e-3, 8e-3),
# )
# ##