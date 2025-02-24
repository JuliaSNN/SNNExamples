using Plots
using DSP  # For the convolution function

# Define the alpha function
function alpha_function(t, τ)
    if t >= 0
        return (t / τ) * exp(1 - t / τ)
    else
        return 0.0
    end
end

# Define parameters
τ = 10.0  # Time constant in ms
bin_width = 1.0  # Bin width in ms
time_range = 0.0:bin_width:100.0  # Time range for the spike train (0 to 100 ms)

# Example spike times in milliseconds
spike_times = [10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

# Create a binary spike train
spike_train = zeros(length(time_range))
for t in spike_times
    index = Int(round(t / bin_width)) + 1
    if index <= length(spike_train)
        spike_train[index] = 1.0
    end
end

# Create the alpha kernel
kernel_length = 100  # Length of the kernel in ms
kernel_time = 0.0:bin_width:kernel_length
alpha_kernel = [alpha_function(t, τ) for t in kernel_time]

# Normalize the kernel (optional, depending on your application)
alpha_kernel ./= sum(alpha_kernel)

# Convolve the spike train with the alpha kernel
instantaneous_rate = conv(spike_train, alpha_kernel)

# Trim the convolved signal to match the original time range
# The convolution result will have length `length(spike_train) + length(alpha_kernel) - 1`
# We trim it to match the original time range
instantaneous_rate = instantaneous_rate[1:length(time_range)]

# Plot the results
plot(time_range, spike_train, label="Spike Train", line=(:stem, :blue))
plot!(time_range, instantaneous_rate, label="Instantaneous Firing Rate", line=(:solid, :red))
xlabel!("Time (ms)")
ylabel!("Activity")
title!("Spike Train and Instantaneous Firing Rate")