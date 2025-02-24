# Load required packages
using Plots
using DSP  # For cross-correlation
using Statistics  # For mean and standard deviation

# Define a function to compute the ISI histogram
function compute_isi_histogram(spike_times; bin_width=1.0)
    # Compute inter-spike intervals (ISI)
    isi = diff(spike_times)
    
    # Create a histogram of ISIs
    isi_hist = histogram(isi, bins=range(0, maximum(isi), step=bin_width),
                         xlabel="Inter-Spike Interval (ms)", ylabel="Frequency",
                         label="ISI Histogram", title="Inter-Spike-Interval Histogram",
                         color=:blue, alpha=0.7)
    return isi, isi_hist
end



    # Plot the cross-correlogram
correlogram_plot = plot(lags, auto_corr, xlabel="Time Lag (ms)", ylabel="Correlation",
                            label="Cross-Correlogram", title="Auto-Correlogram",
                            color=:red, alpha=0.7)

# Example spike times in milliseconds
# spike_times = [10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

# Compute ISI histogram
isi, isi_hist = compute_isi_histogram(spike_times, bin_width=1.0)

# Compute cross-correlogram
lags, auto_corr = compute_correlogram(spike_times, bin_width=1, max_lag=250.0)
correlogram_plot

autocor_plot = autocorrelogram(spike_times, τ=250ms)|> x-> histogram(x,bins=-250:1:250)

plot!(correlogram_plot, autocor_plot)
# Display the plots side by side
auto_corr[51]
plot(isi_hist, correlogram_plot, layout=(1, 2), size=(800, 400))