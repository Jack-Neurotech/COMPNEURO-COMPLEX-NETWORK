
# ==========================================
# STDP LEARNING VISUALIZATION
#
# Runs the branching learning experiment
# and visualizes:
#
#   1. Synaptic weight vs. trial
#   2. Spike-time difference vs. trial
#   3. Spike timing vs. trial
#
# ==========================================

import matplotlib.pyplot as plt

from neuron import Neuron
from network import Network


# ==========================================
# EXPERIMENT PARAMETERS
# ==========================================

NUMBER_OF_TRIALS = 20

INITIAL_WEIGHT_12 = 5.0
INITIAL_WEIGHT_13 = 4.0


# ==========================================
# HISTORY ARRAYS
# ==========================================

weight_12_history = []
weight_13_history = []

delta_t_12_history = []
delta_t_13_history = []

spike_1_history = []
spike_2_history = []
spike_3_history = []


# ==========================================
# PERSISTENT SYNAPTIC WEIGHTS
# ==========================================

current_weight_12 = INITIAL_WEIGHT_12
current_weight_13 = INITIAL_WEIGHT_13


# ==========================================
# RUN LEARNING EXPERIMENT
# ==========================================

for trial in range(
    1,
    NUMBER_OF_TRIALS + 1
):

    # ======================================
    # FRESH NEURONS
    # ======================================

    N1 = Neuron(name="N1")
    N2 = Neuron(name="N2")
    N3 = Neuron(name="N3")


    # ======================================
    # NEW NETWORK
    # ======================================

    network = Network()

    network.add_neuron(N1)
    network.add_neuron(N2)
    network.add_neuron(N3)


    # ======================================
    # RESTORE CURRENT LEARNED WEIGHTS
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
    # REQUIRE SPIKES
    # ======================================

    if not (
        N1.spike_times
        and N2.spike_times
        and N3.spike_times
    ):

        print(
            f"Trial {trial}: "
            "missing spike."
        )

        continue


    # ======================================
    # EXTRACT SPIKE TIMES
    # ======================================

    spike_1 = N1.spike_times[0]
    spike_2 = N2.spike_times[0]
    spike_3 = N3.spike_times[0]


    # ======================================
    # APPLY STDP
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
    # STORE RESULTS
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


# ==========================================
# TRIAL NUMBERS
# ==========================================

trials = range(
    1,
    len(weight_12_history) + 1
)


# ==========================================
# FIGURE 1
# SYNAPTIC WEIGHT VS TRIAL
# ==========================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    trials,
    weight_12_history,
    marker="o",
    label="N1 → N2"
)

plt.plot(
    trials,
    weight_13_history,
    marker="o",
    label="N1 → N3"
)

plt.xlabel(
    "Trial"
)

plt.ylabel(
    "Synaptic Weight"
)

plt.title(
    "Synaptic Weight Changes During STDP Learning"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ==========================================
# FIGURE 2
# SPIKE-TIME DIFFERENCE VS TRIAL
# ==========================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    trials,
    delta_t_12_history,
    marker="o",
    label="N1 → N2"
)

plt.plot(
    trials,
    delta_t_13_history,
    marker="o",
    label="N1 → N3"
)

plt.xlabel(
    "Trial"
)

plt.ylabel(
    "Δt (ms)"
)

plt.title(
    "Pre/Post Spike-Time Difference During Learning"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ==========================================
# FIGURE 3
# SPIKE TIMING VS TRIAL
# ==========================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    trials,
    spike_1_history,
    marker="o",
    label="N1"
)

plt.plot(
    trials,
    spike_2_history,
    marker="o",
    label="N2"
)

plt.plot(
    trials,
    spike_3_history,
    marker="o",
    label="N3"
)

plt.xlabel(
    "Trial"
)

plt.ylabel(
    "Spike Time (ms)"
)

plt.title(
    "Spike Timing Across Learning Trials"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ==========================================
# FINAL NUMERICAL SUMMARY
# ==========================================

print()
print("======================================")
print("LEARNING VISUALIZATION COMPLETE")
print("======================================")

print()
print("Trials:")
print(len(weight_12_history))

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
    "Initial Δt:",
    delta_t_12_history[0],
    "ms"
)
print(
    "Final Δt:",
    delta_t_12_history[-1],
    "ms"
)

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
    "Initial Δt:",
    delta_t_13_history[0],
    "ms"
)
print(
    "Final Δt:",
    delta_t_13_history[-1],
    "ms"
)
