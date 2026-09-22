
# ==========================================
# BRANCHING NETWORK
# REPEATED STDP LEARNING EXPERIMENT
#
#             ┌──→ N2
#             │
# N1 ─────────┤
#             │
#             └──→ N3
#
# Neuron state resets each trial.
# Synaptic weights persist between trials.
# ==========================================

from neuron import Neuron
from synapse import Synapse
from network import Network


# ==========================================
# EXPERIMENT PARAMETERS
# ==========================================

NUMBER_OF_TRIALS = 20

INITIAL_WEIGHT_12 = 5.0
INITIAL_WEIGHT_13 = 4.0


# ==========================================
# CREATE PERSISTENT SYNAPSES
# ==========================================
#
# The neuron objects will be recreated every
# trial.
#
# The synaptic weights, however, persist.
#
# ==========================================

synapse_12 = None
synapse_13 = None


# ==========================================
# HISTORY
# ==========================================

weight_12_history = []
weight_13_history = []

delta_t_12_history = []
delta_t_13_history = []

spike_1_history = []
spike_2_history = []
spike_3_history = []


# ==========================================
# INITIAL WEIGHTS
# ==========================================

current_weight_12 = INITIAL_WEIGHT_12
current_weight_13 = INITIAL_WEIGHT_13


# ==========================================
# RUN LEARNING TRIALS
# ==========================================

for trial in range(
    1,
    NUMBER_OF_TRIALS + 1
):

    # ======================================
    # CREATE FRESH NEURONS
    # ======================================

    N1 = Neuron(
        name="N1"
    )

    N2 = Neuron(
        name="N2"
    )

    N3 = Neuron(
        name="N3"
    )


    # ======================================
    # CREATE FRESH NETWORK
    # ======================================

    network = Network()

    network.add_neuron(N1)
    network.add_neuron(N2)
    network.add_neuron(N3)


    # ======================================
    # CREATE SYNAPSES USING CURRENT WEIGHTS
    # ======================================

    synapse_12 = network.connect(
        pre_neuron=N1,
        post_neuron=N2,
        weight=current_weight_12
    )

    synapse_13 = network.connect(
        pre_neuron=N1,
        post_neuron=N3,
        weight=current_weight_13
    )


    # ======================================
    # RUN NETWORK
    # ======================================

    network.run()


    # ======================================
    # RECORD SPIKES
    # ======================================

    if not N1.spike_times:

        print(
            f"Trial {trial}: "
            "N1 produced no spike."
        )

        continue

    if not N2.spike_times:

        print(
            f"Trial {trial}: "
            "N2 produced no spike."
        )

        continue

    if not N3.spike_times:

        print(
            f"Trial {trial}: "
            "N3 produced no spike."
        )

        continue


    spike_1 = N1.spike_times[0]
    spike_2 = N2.spike_times[0]
    spike_3 = N3.spike_times[0]


    # ======================================
    # CALCULATE STDP
    # ======================================

    result_12 = synapse_12.apply_stdp(
        pre_spike_time=spike_1,
        post_spike_time=spike_2
    )

    result_13 = synapse_13.apply_stdp(
        pre_spike_time=spike_1,
        post_spike_time=spike_3
    )


    # ======================================
    # UPDATE PERSISTENT WEIGHTS
    # ======================================

    current_weight_12 = (
        result_12.new_weight
    )

    current_weight_13 = (
        result_13.new_weight
    )


    # ======================================
    # STORE HISTORY
    # ======================================

    weight_12_history.append(
        current_weight_12
    )

    weight_13_history.append(
        current_weight_13
    )

    delta_t_12_history.append(
        result_12.delta_t
    )

    delta_t_13_history.append(
        result_13.delta_t
    )

    spike_1_history.append(
        spike_1
    )

    spike_2_history.append(
        spike_2
    )

    spike_3_history.append(
        spike_3
    )


    # ======================================
    # PRINT TRIAL
    # ======================================

    print(
        f"Trial {trial}"
    )

    print(
        f"  N1 spike: {spike_1:.4f} ms"
    )

    print(
        f"  N2 spike: {spike_2:.4f} ms"
    )

    print(
        f"  N3 spike: {spike_3:.4f} ms"
    )

    print(
        f"  N1 → N2 Δt: "
        f"{result_12.delta_t:.6f} ms"
    )

    print(
        f"  N1 → N3 Δt: "
        f"{result_13.delta_t:.6f} ms"
    )

    print(
        f"  N1 → N2 weight: "
        f"{result_12.old_weight:.6f}"
        f" → "
        f"{result_12.new_weight:.6f}"
    )

    print(
        f"  N1 → N3 weight: "
        f"{result_13.old_weight:.6f}"
        f" → "
        f"{result_13.new_weight:.6f}"
    )

    print()


# ==========================================
# FINAL RESULTS
# ==========================================

print()
print("======================================")
print("LEARNING EXPERIMENT COMPLETE")
print("======================================")


if weight_12_history:

    print()
    print("N1 → N2")

    print(
        "Initial weight:",
        INITIAL_WEIGHT_12
    )

    print(
        "Final weight:",
        weight_12_history[-1]
    )

    print(
        "Total change:",
        weight_12_history[-1]
        - INITIAL_WEIGHT_12
    )


if weight_13_history:

    print()
    print("N1 → N3")

    print(
        "Initial weight:",
        INITIAL_WEIGHT_13
    )

    print(
        "Final weight:",
        weight_13_history[-1]
    )

    print(
        "Total change:",
        weight_13_history[-1]
        - INITIAL_WEIGHT_13
    )


# ==========================================
# EXPERIMENT SUMMARY
# ==========================================

print()
print("======================================")
print("EXPERIMENT SUMMARY")
print("======================================")

print()
print("Trials completed:")

print(
    len(weight_12_history)
)

print()
print("N1 → N2 weight history:")

print(
    weight_12_history
)

print()
print("N1 → N3 weight history:")

print(
    weight_13_history
)

print()
print("N1 → N2 Δt history:")

print(
    delta_t_12_history
)

print()
print("N1 → N3 Δt history:")

print(
    delta_t_13_history
)
