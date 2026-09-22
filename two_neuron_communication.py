# ==========================================
# TWO-NEURON COMMUNICATION + STDP
# N1 → SYNAPSE → N2
# ==========================================

from neuron_Complex_network import (
    times,
    voltages,
    V_threshold
)

from spike_detection import detect_spikes

from synapse import Synapse

from network_neuron import simulate_neuron


# ==========================================
# STEP 1
# DETECT NEURON 1 SPIKES
# ==========================================

neuron_1_spikes = detect_spikes(
    times,
    voltages,
    threshold=V_threshold
)

print("======================================")
print("NEURON 1")
print("======================================")

print("Neuron 1 spikes:")
print(neuron_1_spikes)


# ==========================================
# STEP 2
# CREATE THE SYNAPSE
# ==========================================

synapse_1_to_2 = Synapse(
    pre_neuron="Neuron 1",
    post_neuron="Neuron 2",
    weight=5.0
)

print()
print("======================================")
print("SYNAPSE")
print("======================================")

print("Initial synaptic weight:")
print(synapse_1_to_2.weight)


# ==========================================
# STEP 3
# TRANSMIT N1 SPIKES
# ==========================================

synaptic_events = []

for spike_time in neuron_1_spikes:

    transmission = synapse_1_to_2.transmit(
        spike_time
    )

    synaptic_events.append(
        (
            transmission["spike_time"],
            transmission["signal"]
        )
    )


print()
print("======================================")
print("SYNAPTIC TRANSMISSION")
print("======================================")

print("Synaptic events:")
print(synaptic_events)


# ==========================================
# STEP 4
# SIMULATE NEURON 2
# ==========================================

neuron_2_times, neuron_2_voltages = simulate_neuron(
    synaptic_events
)


# ==========================================
# STEP 5
# DETECT NEURON 2 SPIKES
# ==========================================

neuron_2_spikes = detect_spikes(
    neuron_2_times,
    neuron_2_voltages,
    threshold=V_threshold
)


print()
print("======================================")
print("NEURON 2")
print("======================================")

print("Neuron 2 spikes:")
print(neuron_2_spikes)


# ==========================================
# STEP 6
# APPLY STDP
# ==========================================

if neuron_1_spikes and neuron_2_spikes:

    pre_spike_time = neuron_1_spikes[0]

    post_spike_time = neuron_2_spikes[0]

    stdp_result = synapse_1_to_2.apply_stdp(
        pre_spike_time=pre_spike_time,
        post_spike_time=post_spike_time
    )


    print()
    print("======================================")
    print("STDP")
    print("======================================")

    print("Pre-synaptic spike:")
    print(pre_spike_time, "ms")

    print("Post-synaptic spike:")
    print(post_spike_time, "ms")

    print("Delta t:")
    print(stdp_result["delta_t"], "ms")

    print("Delta w:")
    print(stdp_result["delta_w"])

    print("New synaptic weight:")
    print(stdp_result["new_weight"])


else:

    print()
    print("======================================")
    print("STDP")
    print("======================================")

    print(
        "STDP was not applied because "
        "one or both neurons did not produce a spike."
    )