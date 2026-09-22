
# ==========================================
# BRANCHING NEURAL NETWORK
# API-SAFE STDP INTEGRATION
#
#             ┌──→ N2
#             │
# N1 ─────────┤
#             │
#             └──→ N3
#
# ==========================================

from neuron import Neuron
from network import Network


# ==========================================
# CREATE NEURONS
# ==========================================

N1 = Neuron(name="N1")
N2 = Neuron(name="N2")
N3 = Neuron(name="N3")


# ==========================================
# CREATE NETWORK
# ==========================================

network = Network()

network.add_neuron(N1)
network.add_neuron(N2)
network.add_neuron(N3)


# ==========================================
# CREATE SYNAPSES
# ==========================================

synapse_12 = network.connect(
    pre_neuron=N1,
    post_neuron=N2,
    weight=5.0
)

synapse_13 = network.connect(
    pre_neuron=N1,
    post_neuron=N3,
    weight=4.0
)


# ==========================================
# STORE INITIAL WEIGHTS
# ==========================================

initial_weight_12 = synapse_12.weight
initial_weight_13 = synapse_13.weight


# ==========================================
# RUN NETWORK
# ==========================================

network.run()


# ==========================================
# NETWORK TOPOLOGY
# ==========================================

print()
print("======================================")
print("BRANCHING NEURAL NETWORK")
print("======================================")

print()
print("NETWORK TOPOLOGY")

print("N1")
print("|---> N2")
print("|---> N3")


# ==========================================
# SPIKE RESULTS
# ==========================================

print()
print("======================================")
print("SPIKE RESULTS")
print("======================================")

print()
print("N1 spikes:")
print(N1.spike_times)

print()
print("N2 spikes:")
print(N2.spike_times)

print()
print("N3 spikes:")
print(N3.spike_times)


# ==========================================
# SPIKE-TIMING RELATIONSHIPS
# ==========================================

print()
print("======================================")
print("SPIKE-TIMING RELATIONSHIPS")
print("======================================")


# ==========================================
# N1 → N2
# ==========================================

stdp_12 = None

if N1.spike_times and N2.spike_times:

    pre_spike_12 = N1.spike_times[0]
    post_spike_12 = N2.spike_times[0]

    delta_t_12 = (
        post_spike_12
        - pre_spike_12
    )

    print()
    print("N1 → N2 Δt:")
    print(delta_t_12, "ms")

    # --------------------------------------
    # API-SAFE STDP CALL
    # --------------------------------------

    stdp_12 = synapse_12.apply_stdp(
        pre_spike_time=pre_spike_12,
        post_spike_time=post_spike_12
    )


# ==========================================
# N1 → N3
# ==========================================

stdp_13 = None

if N1.spike_times and N3.spike_times:

    pre_spike_13 = N1.spike_times[0]
    post_spike_13 = N3.spike_times[0]

    delta_t_13 = (
        post_spike_13
        - pre_spike_13
    )

    print()
    print("N1 → N3 Δt:")
    print(delta_t_13, "ms")

    # --------------------------------------
    # API-SAFE STDP CALL
    # --------------------------------------

    stdp_13 = synapse_13.apply_stdp(
        pre_spike_time=pre_spike_13,
        post_spike_time=post_spike_13
    )


# ==========================================
# STDP RESULTS
# ==========================================

print()
print("======================================")
print("STDP RESULTS")
print("======================================")


# ==========================================
# N1 → N2 STDP
# ==========================================

print()
print("N1 → N2")

if stdp_12 is not None:

    print()
    print("Pre-synaptic spike:")
    print(stdp_12.pre_spike_time, "ms")

    print()
    print("Post-synaptic spike:")
    print(stdp_12.post_spike_time, "ms")

    print()
    print("Delta t:")
    print(stdp_12.delta_t, "ms")

    print()
    print("Delta w:")
    print(stdp_12.delta_w)

    print()
    print("Old weight:")
    print(stdp_12.old_weight)

    print()
    print("New weight:")
    print(stdp_12.new_weight)

else:

    print("No STDP update.")


# ==========================================
# N1 → N3 STDP
# ==========================================

print()
print("N1 → N3")

if stdp_13 is not None:

    print()
    print("Pre-synaptic spike:")
    print(stdp_13.pre_spike_time, "ms")

    print()
    print("Post-synaptic spike:")
    print(stdp_13.post_spike_time, "ms")

    print()
    print("Delta t:")
    print(stdp_13.delta_t, "ms")

    print()
    print("Delta w:")
    print(stdp_13.delta_w)

    print()
    print("Old weight:")
    print(stdp_13.old_weight)

    print()
    print("New weight:")
    print(stdp_13.new_weight)

else:

    print("No STDP update.")


# ==========================================
# FINAL NETWORK STATE
# ==========================================

print()
print("======================================")
print("FINAL NETWORK STATE")
print("======================================")

print()
print("N1 → N2:")
print(
    initial_weight_12,
    "→",
    synapse_12.weight
)

print()
print("N1 → N3:")
print(
    initial_weight_13,
    "→",
    synapse_13.weight
)

