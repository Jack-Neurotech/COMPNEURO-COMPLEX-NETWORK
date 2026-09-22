
# ==========================================
# CHAIN NETWORK
#
# N1 → N2 → N3
#
# N1 generates the initial spike.
# N1 transmits to N2.
# N2 generates its own spike.
# N2 transmits to N3.
#
# STDP is applied independently to:
#
#   N1 → N2
#   N2 → N3
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
# CREATE CHAIN
# ==========================================

synapse_12 = network.connect(
    pre_neuron=N1,
    post_neuron=N2,
    weight=5.0
)

synapse_23 = network.connect(
    pre_neuron=N2,
    post_neuron=N3,
    weight=5.0
)


# ==========================================
# INITIAL WEIGHTS
# ==========================================

initial_weight_12 = synapse_12.weight
initial_weight_23 = synapse_23.weight


# ==========================================
# RUN NETWORK
# ==========================================

network.run()


# ==========================================
# CHECK SPIKES
# ==========================================

if not N1.spike_times:

    raise RuntimeError(
        "N1 did not produce a spike."
    )

if not N2.spike_times:

    raise RuntimeError(
        "N2 did not produce a spike."
    )

if not N3.spike_times:

    raise RuntimeError(
        "N3 did not produce a spike."
    )


# ==========================================
# EXTRACT FIRST SPIKE
# ==========================================

spike_1 = N1.spike_times[0]
spike_2 = N2.spike_times[0]
spike_3 = N3.spike_times[0]


# ==========================================
# CALCULATE SPIKE-TIME DIFFERENCES
# ==========================================

delta_t_12 = (
    spike_2
    - spike_1
)

delta_t_23 = (
    spike_3
    - spike_2
)


# ==========================================
# APPLY STDP
# ==========================================

result_12 = synapse_12.apply_stdp(
    pre_spike_time=spike_1,
    post_spike_time=spike_2
)

result_23 = synapse_23.apply_stdp(
    pre_spike_time=spike_2,
    post_spike_time=spike_3
)


# ==========================================
# OUTPUT
# ==========================================

print()
print("======================================")
print("CHAIN NETWORK")
print("======================================")

print()
print("Network:")
print("N1 → N2 → N3")

print()
print("======================================")
print("SPIKE TIMES")
print("======================================")

print()
print("N1 spike:")
print(
    f"{spike_1:.4f} ms"
)

print()
print("N2 spike:")
print(
    f"{spike_2:.4f} ms"
)

print()
print("N3 spike:")
print(
    f"{spike_3:.4f} ms"
)


# ==========================================
# SYNAPTIC TIMING
# ==========================================

print()
print("======================================")
print("SYNAPTIC TIMING")
print("======================================")

print()
print("N1 → N2 Δt:")
print(
    f"{delta_t_12:.6f} ms"
)

print()
print("N2 → N3 Δt:")
print(
    f"{delta_t_23:.6f} ms"
)


# ==========================================
# STDP RESULTS
# ==========================================

print()
print("======================================")
print("STDP")
print("======================================")

print()
print("N1 → N2")

print(
    "Weight:",
    f"{initial_weight_12:.6f}",
    "→",
    f"{result_12.new_weight:.6f}"
)

print(
    "Δw:",
    f"{result_12.delta_w:.6f}"
)

print()
print("N2 → N3")

print(
    "Weight:",
    f"{initial_weight_23:.6f}",
    "→",
    f"{result_23.new_weight:.6f}"
)

print(
    "Δw:",
    f"{result_23.delta_w:.6f}"
)


# ==========================================
# PROPAGATION DELAY
# ==========================================

total_delay = (
    spike_3
    - spike_1
)

print()
print("======================================")
print("PROPAGATION")
print("======================================")

print()
print("N1 → N3 total propagation delay:")

print(
    f"{total_delay:.6f} ms"
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print()
print("======================================")
print("CHAIN EXPERIMENT COMPLETE")
print("======================================")

print()
print("Spike propagation:")

print(
    f"N1 ({spike_1:.4f})"
    " → "
    f"N2 ({spike_2:.4f})"
    " → "
    f"N3 ({spike_3:.4f})"
)


