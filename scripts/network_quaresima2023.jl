using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Distributions

## Define the network
network = let
    # Number of neurons in the network
    NE = 1000
    NI1 = 175
    NI2 = 325
    # Define neurons and synapses in the network
	# proximal_distal = [(150um, 400um), (150um, 400um)], defines the dendrite dimensions later used in create_dendrite
    E = SNN.Tripod(proximal_distal...;
        N = NE,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma), # defines glutamaterbic and gabaergic receptors in the soma
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        NMDA = SNN.EyalNMDA,
        param = SNN.AdExSoma(Vr = -55mV, Vt = -50mV),
    )

    # Define interneurons I1 and I2
    I1 = SNN.IF(; N = NI1, param = SNNUtils.PVDuarte)
    I2 = SNN.IF(; N = NI2, param = SNNUtils.SSTDuarte)

    # Define synaptic interactions between neurons and interneurons
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge, p = 0.2, σ = 5.27)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge, p = 0.2, σ = 5.27)

    I1_to_E = SNN.CompartmentSynapse(
        I1,
        E,
        :s,
        :inh,
        p = 0.2,
        σ = 15.8,
        param = SNNUtils.quaresima2023.iSTDP_rate,
    )

    I2_to_E_d1 = SNN.CompartmentSynapse( # what learning rate should I use?
        I2,
        E,
        :d1,
        :inh,
        p = 0.2,
        σ = 15.8,
        param = SNNUtils.quaresima2023.iSTDP_potential,
    )


    I2_to_E_d2 = SNN.CompartmentSynapse( # what learning rate should I use?
        I2,
        E,
        :d2,
        :inh,
        p = 0.2,
        σ = 15.8,
        param = SNN.no_STDPParameter(),
    )

    # Define recurrent connections in the network
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi, p = 0.2, σ = 16.2)
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi, p = 0.2, σ = 16.2)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi, p = 0.2, σ = 1.47)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi, p = 0.2, σ = 0.83)

    E_to_E1_d1 = SNN.CompartmentSynapse(
        E,
        E,
        :d1,
        :exc,
        p = 0.2,
        σ = 10.78,
        param = SNNUtils.quaresima2023.vstdp,
    )

    E_to_E2_d2 = SNN.CompartmentSynapse(
        E,
        E,
        :d2,
        :exc,
        p = 0.2,
        σ = 10.78,
        param = SNNUtils.quaresima2023.vstdp,
    )

    # Define normalization
    recurrent_norm =
    [
        SNN.SynapseNormalization(NE, [E_to_E1_d1], param = SNN.MultiplicativeNorm(τ = 20ms)),
        SNN.SynapseNormalization(NE, [E_to_E2_d2], param = SNN.MultiplicativeNorm(τ = 20ms))
    ]

    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E_d1 I2_to_E_d2 I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E1_d1 E_to_E2_d2 norm1=recurrent_norm[1] norm2=recurrent_norm[2])
    
	# Return the network as a tuple
    (pop = pop, syn = syn)
end

function ramp(time::Float32)
    if time > 5500ms && time < 5700ms
        return 1000Hz
    else
        return 0.0
    end
end

stimuli = Dict(
    ## Background noise
    "noise_s"  => SNN.PoissonStimulus(network.pop.E, :h_s, x->1000Hz, cells=:ALL, σ=10.f0),
    "stim1_d1" => SNN.PoissonStimulus(network.pop.E, :h_d1, ramp, σ=10.f0),
)


model = SNN.merge_models(network, stim=stimuli)
SNN.train!(model=model, duration= 10s, pbar=true)

##
dictionary = Dict("AB"=>["A", "B"], "CD"=>["C", "D"])
duration = Dict("A"=>40, "B"=>60, "C"=>45, "D"=>40, "_"=>100)
config = (seq_length=100, silence=1, dictionary=dictionary, ph_duration=duration)
seq = generate_sequence(config)

intervals = word_intervals(1, seq)
print(intervals[3])

