# ==========================================
# CHAIN LEARNING EXPERIMENT
# N1 → N2 → N3
# REPEATED STDP LEARNING
# ==========================================

import matplotlib.pyplot as plt

from neuron import Neuron
from network import Network


# ==========================================
# EXPERIMENT PARAMETERS
# ==========================================

NUMBER_OF_TRIALS = 20

INITIAL_WEIGHT_12 = 5.0
INITIAL_WEIGHT_23 = 5.0


# ==========================================
# PERSISTENT SYNAPTIC WEIGHTS
# ==========================================

current_weight_12 = INITIAL_WEIGHT_12
current_weight_23 = INITIAL_WEIGHT_23


# ==========================================
# STORE LEARNING HISTORY
# ==========================================

weight_12_history = []
weight_23_history = []

delta_t_12_history = []
delta_t_23_history = []

n1_spike_history = []
n2_spike_history = []
n3_spike_history = []


# ==========================================
# RUN REPEATED LEARNING TRIALS
# ==========================================

for trial in range(1, NUMBER_OF_TRIALS + 1):

    # --------------------------------------
    # CREATE FRESH NEURONS
    # --------------------------------------

    N1 = Neuron(name="N1")
    N2 = Neuron(name="N2")
    N3 = Neuron(name="N3")

    # --------------------------------------
    # CREATE FRESH NETWORK
    # --------------------------------------

    network = Network()

    network.add_neuron(N1)
    network.add_neuron(N2)
    network.add_neuron(N3)

    # --------------------------------------
    # CONNECT THE CHAIN
    # --------------------------------------

    synapse_12 = network.connect(
        pre_neuron=N1,
        post_neuron=N2,
        weight=current_weight_12
    )

    synapse_23 = network.connect(
        pre_neuron=N2,
        post_neuron=N3,
        weight=current_weight_23
    )

    # --------------------------------------
    # RUN NETWORK
    # --------------------------------------

    network.run()

    # --------------------------------------
    # VERIFY SPIKES
    # --------------------------------------

    if not N1.spike_times:
        raise RuntimeError("N1 did not produce a spike.")

    if not N2.spike_times:
        raise RuntimeError("N2 did not produce a spike.")

    if not N3.spike_times:
        raise RuntimeError("N3 did not produce a spike.")

    # --------------------------------------
    # GET FIRST SPIKES
    # --------------------------------------

    n1_spike = N1.spike_times[0]
    n2_spike = N2.spike_times[0]
    n3_spike = N3.spike_times[0]

    # --------------------------------------
    # CALCULATE SPIKE-TIMING DIFFERENCES
    # --------------------------------------

    delta_t_12 = n2_spike - n1_spike
    delta_t_23 = n3_spike - n2_spike

    # --------------------------------------
    # APPLY STDP
    # --------------------------------------

    result_12 = synapse_12.apply_stdp(
        pre_spike_time=n1_spike,
        post_spike_time=n2_spike
    )

    result_23 = synapse_23.apply_stdp(
        pre_spike_time=n2_spike,
        post_spike_time=n3_spike
    )

    # --------------------------------------
    # PERSIST NEW WEIGHTS
    # --------------------------------------

    current_weight_12 = result_12.new_weight
    current_weight_23 = result_23.new_weight

    # --------------------------------------
    # STORE HISTORY
    # --------------------------------------

    weight_12_history.append(current_weight_12)
    weight_23_history.append(current_weight_23)

    delta_t_12_history.append(delta_t_12)
    delta_t_23_history.append(delta_t_23)

    n1_spike_history.append(n1_spike)
    n2_spike_history.append(n2_spike)
    n3_spike_history.append(n3_spike)

    # --------------------------------------
    # PRINT TRIAL
    # --------------------------------------

    print()
    print("======================================")
    print(f"TRIAL {trial}")
    print("======================================")

    print(f"N1 spike: {n1_spike:.4f} ms")
    print(f"N2 spike: {n2_spike:.4f} ms")
    print(f"N3 spike: {n3_spike:.4f} ms")

    print()
    print(f"N1 → N2 Δt: {delta_t_12:.6f} ms")
    print(f"N2 → N3 Δt: {delta_t_23:.6f} ms")

    print()
    print(
        f"N1 → N2 weight: "
        f"{result_12.old_weight:.6f} → "
        f"{result_12.new_weight:.6f}"
    )

    print(
        f"N2 → N3 weight: "
        f"{result_23.old_weight:.6f} → "
        f"{result_23.new_weight:.6f}"
    )


# ==========================================
# FINAL SUMMARY
# ==========================================

print()
print("======================================")
print("CHAIN LEARNING COMPLETE")
print("======================================")

print()
print("N1 → N2")
print(f"Initial weight: {INITIAL_WEIGHT_12:.6f}")
print(f"Final weight:   {current_weight_12:.6f}")
print(
    f"Total change:   "
    f"{current_weight_12 - INITIAL_WEIGHT_12:.6f}"
)

print()
print("N2 → N3")
print(f"Initial weight: {INITIAL_WEIGHT_23:.6f}")
print(f"Final weight:   {current_weight_23:.6f}")
print(
    f"Total change:   "
    f"{current_weight_23 - INITIAL_WEIGHT_23:.6f}"
)

print()
print("Timing")

print(
    f"N1 → N2 Δt: "
    f"{delta_t_12_history[0]:.6f} → "
    f"{delta_t_12_history[-1]:.6f} ms"
)

print(
    f"N2 → N3 Δt: "
    f"{delta_t_23_history[0]:.6f} → "
    f"{delta_t_23_history[-1]:.6f} ms"
)


# ==========================================
# VISUALIZATION 1
# SYNAPTIC WEIGHTS
# ==========================================

trials = list(range(1, NUMBER_OF_TRIALS + 1))

plt.figure(figsize=(10, 6))

plt.plot(
    trials,
    weight_12_history,
    marker="o",
    label="N1 → N2"
)

plt.plot(
    trials,
    weight_23_history,
    marker="o",
    label="N2 → N3"
)

plt.xlabel("Trial")
plt.ylabel("Synaptic Weight")
plt.title("Synaptic Weight Learning Across the Chain")
plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# VISUALIZATION 2
# SPIKE-TIMING DIFFERENCE
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    trials,
    delta_t_12_history,
    marker="o",
    label="N1 → N2"
)

plt.plot(
    trials,
    delta_t_23_history,
    marker="o",
    label="N2 → N3"
)

plt.xlabel("Trial")
plt.ylabel("Δt (ms)")
plt.title("Spike-Timing Changes Across the Chain")
plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# VISUALIZATION 3
# SPIKE TIMES
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    trials,
    n1_spike_history,
    marker="o",
    label="N1"
)

plt.plot(
    trials,
    n2_spike_history,
    marker="o",
    label="N2"
)

plt.plot(
    trials,
    n3_spike_history,
    marker="o",
    label="N3"
)

plt.xlabel("Trial")
plt.ylabel("Spike Time (ms)")
plt.title("Spike Timing Across the Neural Chain")
plt.legend()
plt.grid(True)

plt.show()