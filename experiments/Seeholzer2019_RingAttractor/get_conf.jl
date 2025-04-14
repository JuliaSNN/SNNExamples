using CSV
using DataFrames

# Define the named tuple structure
config = (
    E_to_I = 1.29,
    I_to_I = 2.7,
    I_to_E = 1.8,
    σ_w = 0.38,
    w_max = 0.25,
    STPparam = (
        τD = 150ms, # τx
        τF = 650ms, # τu
        U = 0.5,
    ),
    NE = 1600,
    ΔT = 1s,
    input_neurons = [400:500],
    sparsity = 1,
)

# Function to read the CSV and create the configuration
# Example usage

println(new_config)
