# ==========================================
# SPIKE DETECTION
# ==========================================
#
# This module takes the voltage output from
# the existing neuron model and identifies
# when the membrane voltage crosses threshold.
#
# IMPORTANT:
# This does NOT create a new neuron model.
# The six-piecewise neuron remains the source
# of truth.
# ==========================================


# ==========================================
# SPIKE DETECTION FUNCTION
# ==========================================

def detect_spikes(times, voltages, threshold=-55.0):

    # Store the times at which spikes occur.
    spike_times = []


    # ======================================
    # CHECK EACH VOLTAGE SAMPLE
    # ======================================

    for i in range(1, len(voltages)):

        previous_voltage = voltages[i - 1]
        current_voltage = voltages[i]


        # ==================================
        # THRESHOLD CROSSING
        # ==================================
        #
        # A spike is detected when voltage
        # moves from below threshold to
        # threshold or above it.
        #
        # This prevents the neuron from being
        # counted as "spiking" at every sample
        # while it remains above threshold.
        # ==================================

        if (
            previous_voltage < threshold
            and current_voltage >= threshold
        ):

            spike_times.append(times[i])


    return spike_times


# ==========================================
# CONNECT TO THE ACTUAL NEURON
# ==========================================

from neuron_Complex_network import (
    times,
    voltages,
    V_threshold
)


# ==========================================
# DETECT SPIKES
# ==========================================

spike_times = detect_spikes(
    times,
    voltages,
    threshold=V_threshold
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("Detected spike times:")
print(spike_times)

print()
print("Number of spikes:")
print(len(spike_times))