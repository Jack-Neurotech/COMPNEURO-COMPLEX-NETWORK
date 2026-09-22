# ==========================================
# TWO-NEURON NETWORK
# ==========================================
#
# This is the first network-level experiment.
#
# Neuron 1:
#     produces a spike
#
# Synapse:
#     receives the spike and applies a weight
#
# Neuron 2:
#     receives the synaptic signal
#
# At this stage we are testing the
# communication architecture.
# ==========================================


# ==========================================
# IMPORT THE EXISTING COMPONENTS
# ==========================================

from neuron_Complex_network import (
    times,
    voltages,
    V_threshold
)

from spike_detection import detect_spikes

from synapse import Synapse


# ==========================================
# NEURON 1 — SPIKE DETECTION
# ==========================================

neuron_1_spikes = detect_spikes(
    times,
    voltages,
    threshold=V_threshold
)


# ==========================================
# CREATE SYNAPSE
# ==========================================
#
# Neuron 1 is presynaptic.
# Neuron 2 is postsynaptic.
#
# The initial synaptic weight is 0.5.
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

    transmissions.append(transmission)


# ==========================================
# DISPLAY NETWORK ACTIVITY
# ==========================================

print()
print("==========================================")
print("TWO-NEURON NETWORK")
print("==========================================")

print()

print("Neuron 1 spike times:")
print(neuron_1_spikes)

print()

print("Synaptic transmissions:")

for transmission in transmissions:

    print(transmission)

print()

print("Synaptic weight:")
print(synapse_1_to_2.weight)