using Distributions

function ballstick_WM(;params, name, NE, STDP, kwargs...)
    @unpack exc, pv, sst, plasticity, connectivity=params
    @unpack  noise_params, inh_ratio = params
    exc = quaresima2022
    # pv = duarte2019.PV
    # sst = duarte2019.SST
    # @unpack connectivity, plasticity = quaresima2023
    @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    # Number of neurons in the network
    NI1 = round(Int,NE * inh_ratio.ni1)
    NI2 = round(Int,NE * inh_ratio.ni2)
    # Import models parameters
    # Define interneurons I1 and I2
    # @unpack dends, NMDA, param, soma_syn, dend_syn = exc
    E = SNN.BallAndStickHet(; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="ExcBallStick")
    I1 = SNN.IF(; N = NI1, param = pv, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = sst, name="I2_sst")
    # Define synaptic interactions between neurons and interneurons
    E_to_E = SNN.SpikingSynapse(E, E, :he, :d ; connectivity.E_to_Ed..., param= plasticity.vstdp)
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.E_to_If...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.E_to_Is...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.If_to_E...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.If_to_If...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.Is_to_If...)
    I2_to_I2 = SNN.SpikingSynapse(I2, I2, :gi; connectivity.Is_to_Is...)
    I2_to_E = SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.Is_to_Ed...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.If_to_Is...)
    # Define normalization
    norm = SNN.SynapseNormalization(NE, [E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    # background noise
    @unpack exc_soma, exc_dend, inh1, inh2= noise_params
    stimuli = Dict(
        :s   => SNN.PoissonStimulus(E,  :he_s; exc_soma... ),
        :d   => SNN.PoissonStimulus(E,  :he_d; exc_dend... ),
        :i1  => SNN.PoissonStimulus(I1, :ge;   inh1...  ),
        :i2  => SNN.PoissonStimulus(I2, :ge;   inh2...  )
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    # Return the network as a model
    if !STDP
        syn.E_to_E.param.active[1] = false
    end
    merge_models(pop, syn, noise=stimuli, name=name)
end
