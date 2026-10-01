
# ==========================================
# TWO-NEURON NETWORK
# ==========================================
#
# Network-level communication experiment.
#
# Neuron 1:
#     produces an action potential
#
# Synapse:
#     receives the spike
#     applies a synaptic weight
#
# Neuron 2:
#     receives the synaptic signal
#
# This experiment tests the basic
# communication architecture.
# ==========================================


# ==========================================
# IMPORT NUMPY
# ==========================================

import numpy as np


# ==========================================
# IMPORT THE EXISTING NEURON MODEL
# ==========================================
#
# The neuron model now exposes its
# simulation through:
#
#     simulate_neuron()
#
# It returns:
#
#     times
#     voltages
#     spike_times
# ==========================================

from neuron_Complex_network import (
    simulate_neuron,
    V_threshold
)


# ==========================================
# IMPORT SPIKE DETECTION
# ==========================================

from spike_detection import detect_spikes


# ==========================================
# IMPORT SYNAPSE
# ==========================================

from synapse import Synapse


# ==========================================
# NEURON 1
# ==========================================
#
# Run the existing deterministic neuron.
#
# intrinsic_input=True means Neuron 1
# receives its normal intrinsic input.
# ==========================================

neuron_1_times, neuron_1_voltages, neuron_1_spikes = (
    simulate_neuron(
        intrinsic_input=True
    )
)


# ==========================================
# VERIFY NEURON 1 SPIKES
# ==========================================
#
# The neuron model already provides
# spike_times.
#
# We also run the existing spike detector
# against the voltage trace so this
# experiment continues to use the
# project's spike-detection component.
# ==========================================

detected_spikes = detect_spikes(
    neuron_1_times,
    neuron_1_voltages,
    threshold=V_threshold
)


# ==========================================
# USE DETECTED SPIKES
# ==========================================

neuron_1_spikes = np.asarray(
    detected_spikes,
    dtype=float
)


# ==========================================
# CHECK FOR SPIKE
# ==========================================

if neuron_1_spikes.size == 0:

    raise RuntimeError(
        "Neuron 1 did not produce a spike."
    )


# ==========================================
# CREATE SYNAPSE
# ==========================================
#
# Neuron 1 is presynaptic.
#
# Neuron 2 is postsynaptic.
#
# Initial synaptic weight:
#
#     0.5
# ==========================================

synapse_1_to_2 = Synapse(
    pre_neuron="Neuron 1",
    post_neuron="Neuron 2",
    weight=0.5
)


# ==========================================
# TRANSMIT NEURON 1 SPIKES
# ==========================================

transmissions = []


for spike_time in neuron_1_spikes:

    transmission = synapse_1_to_2.transmit(
        spike_time
    )

    transmissions.append(
        transmission
    )


# ==========================================
# DISPLAY NETWORK ACTIVITY
# ==========================================

print()

print("==========================================")

print("TWO-NEURON NETWORK")

print("==========================================")


# ==========================================
# NEURON 1 RESULTS
# ==========================================

print()

print("Neuron 1 spike times:")

print(
    neuron_1_spikes
)


print()

print("Number of Neuron 1 spikes:")

print(
    neuron_1_spikes.size
)


# ==========================================
# SYNAPTIC TRANSMISSION
# ==========================================

print()

print("Synaptic transmissions:")


for transmission in transmissions:

    print(
        transmission
    )


# ==========================================
# SYNAPTIC WEIGHT
# ==========================================

print()

print("Synaptic weight:")

print(
    synapse_1_to_2.weight
)


# ==========================================
# EXPERIMENT COMPLETE
# ==========================================

print()

print("==========================================")

print("TWO-NEURON NETWORK COMPLETE")

print("==========================================")
