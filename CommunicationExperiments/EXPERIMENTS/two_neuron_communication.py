
# =========================================
# TWO-NEURON COMMUNICATION EXPERIMENT
#
# N1 → SYNAPSE → N2
#
# N1:
#     Intrinsically driven neuron
#
# N2:
#     Synaptically driven neuron
#
# SYNAPSE:
#     Transmits N1 spike to N2
#
# STDP:
#     Compares N1 and N2 spike timing
#     and updates the synaptic weight.
# ==========================================


# ==========================================
# IMPORTS
# ==========================================

from neuron_Complex_network import (
    simulate_neuron
)

from synapse import (
    Synapse
)


# ==========================================
# NEURON 1
# INTRINSICALLY DRIVEN
# ==========================================

print()
print("======================================")
print("NEURON 1")
print("======================================")


(
    neuron_1_times,
    neuron_1_voltages,
    neuron_1_spikes
) = simulate_neuron(
    intrinsic_input=True
)


# ==========================================
# DISPLAY NEURON 1 SPIKES
# ==========================================

print()
print("Neuron 1 spikes:")

print(
    neuron_1_spikes
)


print()
print("Number of spikes:")

print(
    len(neuron_1_spikes)
)


# ==========================================
# VERIFY NEURON 1 SPIKE
# ==========================================

if neuron_1_spikes.size == 0:
    raise RuntimeError(
        "Neuron 1 did not produce a spike."
    )


# ==========================================
# FIRST N1 SPIKE
# ==========================================

pre_spike_time = (
    neuron_1_spikes[0]
)


# ==========================================
# CREATE SYNAPSE
# ==========================================

print()
print("======================================")
print("SYNAPSE")
print("======================================")


synapse = Synapse(
    pre_neuron="N1",
    post_neuron="N2",
    weight=5.0
)


print()
print("Pre-synaptic neuron:")

print(
    synapse.pre_neuron
)


print()
print("Post-synaptic neuron:")

print(
    synapse.post_neuron
)


print()
print("Initial synaptic weight:")

print(
    synapse.weight
)


# ==========================================
# SYNAPTIC TRANSMISSION
# ==========================================

print()
print("======================================")
print("SYNAPTIC TRANSMISSION")
print("======================================")


transmission = synapse.transmit(
    spike_time=pre_spike_time
)


print()
print("Transmission:")

print(
    transmission
)


# ==========================================
# CREATE SYNAPTIC EVENT
# ==========================================
#
# The event contains:
#
#     spike time
#     synaptic current
#
# N2 will receive this event.
# ==========================================

synaptic_events = [

    (
        transmission["spike_time"],
        transmission["signal"]
    )

]


print()
print("Synaptic events:")

print(
    synaptic_events
)


# ==========================================
# NEURON 2
# SYNAPTICALLY DRIVEN
# ==========================================

print()
print("======================================")
print("NEURON 2")
print("======================================")


(
    neuron_2_times,
    neuron_2_voltages,
    neuron_2_spikes
) = simulate_neuron(
    synaptic_events=synaptic_events,
    intrinsic_input=False
)


# ==========================================
# DISPLAY NEURON 2 SPIKES
# ==========================================

print()
print("Neuron 2 spikes:")

print(
    neuron_2_spikes
)


print()
print("Number of spikes:")

print(
    len(neuron_2_spikes)
)


# ==========================================
# VERIFY NEURON 2 SPIKE
# ==========================================

if not neuron_2_spikes:

    raise RuntimeError(
        "Neuron 2 did not produce a spike."
    )


# ==========================================
# FIRST N2 SPIKE
# ==========================================

post_spike_time = (
    neuron_2_spikes[0]
)


# ==========================================
# STDP
# ==========================================

print()
print("======================================")
print("STDP")
print("======================================")


stdp_result = synapse.apply_stdp(
    pre_spike_time=pre_spike_time,
    post_spike_time=post_spike_time
)


# ==========================================
# DISPLAY STDP RESULTS
# ==========================================

print()
print("Pre-synaptic spike:")

print(
    stdp_result.pre_spike_time,
    "ms"
)


print()
print("Post-synaptic spike:")

print(
    stdp_result.post_spike_time,
    "ms"
)


print()
print("Delta t:")

print(
    stdp_result.delta_t,
    "ms"
)


print()
print("Delta w:")

print(
    stdp_result.delta_w
)


print()
print("Old synaptic weight:")

print(
    stdp_result.old_weight
)


print()
print("New synaptic weight:")

print(
    stdp_result.new_weight
)


# ==========================================
# FINAL SYNAPTIC STATE
# ==========================================

print()
print("======================================")
print("FINAL SYNAPTIC STATE")
print("======================================")


print()
print("N1 → N2 weight:")

print(
    synapse.weight
)


# ==========================================
# EXPERIMENT COMPLETE
# ==========================================

print()
print("======================================")
print("TWO-NEURON EXPERIMENT COMPLETE")
print("======================================")
