# ============================================================
# PLOT LEARNING CURVES
# ============================================================
#
# Purpose:
# Quantitatively visualize learning in the convergent
# computational neuroscience network.
#
# Network:
#
#       N1 ──────┐
#                ├──→ N3 ──→ N4
#       N2 ──────┘
#
# This experiment uses the existing:
#
#   1. Six-phase neuron model
#   2. Shared global clock
#   3. Synaptic communication
#   4. Convergent network topology
#   5. Persistent synaptic weights
#   6. STDP learning
#
# Four measurements are plotted:
#
#   1. Synaptic weight vs. trial
#   2. Spike time vs. trial
#   3. STDP Δt vs. trial
#   4. Network propagation time vs. trial
#
# Existing project files are NOT modified.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import matplotlib.pyplot as plt

from neuron import Neuron
from shared_clock_network import SharedClockNetwork
from stdp_models import STDPParameters


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

NUM_TRIALS = 20
INITIAL_WEIGHT = 5.0

STDP_PARAMETERS = STDPParameters()


# ============================================================
# CREATE NETWORK
# ============================================================

network = SharedClockNetwork(
    simulation_time=50.0,
    dt=0.01
)

n1 = Neuron(
    name="N1"
)

n2 = Neuron(
    name="N2"
)

n3 = Neuron(
    name="N3"
)

n4 = Neuron(
    name="N4"
)


# ============================================================
# ADD NEURONS
# ============================================================

network.add_neuron(n1)
network.add_neuron(n2)
network.add_neuron(n3)
network.add_neuron(n4)


# ============================================================
# CREATE CONVERGENT NETWORK
# ============================================================
#
#       N1 ──────┐
#                ├──→ N3 ──→ N4
#       N2 ──────┘
#
# N1 and N2 are source neurons.
#
# Their activity converges onto N3.
#
# N3 then drives N4.
# ============================================================

synapse_n1_n3 = network.connect(
    n1,
    n3,
    weight=INITIAL_WEIGHT
)

synapse_n2_n3 = network.connect(
    n2,
    n3,
    weight=INITIAL_WEIGHT
)

synapse_n3_n4 = network.connect(
    n3,
    n4,
    weight=INITIAL_WEIGHT
)


# ============================================================
# HISTORY ARRAYS
# ============================================================

trial_history = []

# Synaptic weights
weight_n1_n3 = []
weight_n2_n3 = []
weight_n3_n4 = []

# Spike times
spike_n1 = []
spike_n2 = []
spike_n3 = []
spike_n4 = []

# STDP timing differences
delta_t_n1_n3 = []
delta_t_n2_n3 = []
delta_t_n3_n4 = []

# Total network propagation
propagation_n1_n4 = []


# ============================================================
# EXPERIMENT HEADER
# ============================================================

print()
print("=" * 70)
print("STDP LEARNING CURVE EXPERIMENT")
print("=" * 70)

print()

print("Network:")
print()
print("N1 ──────┐")
print("         ├──→ N3 ──→ N4")
print("N2 ──────┘")

print()

print(f"Learning trials: {NUM_TRIALS}")
print(f"Initial synaptic weight: {INITIAL_WEIGHT}")

print()


# ============================================================
# RUN LEARNING TRIALS
# ============================================================

for trial in range(
    1,
    NUM_TRIALS + 1
):

    print(
        f"Running trial {trial}/{NUM_TRIALS}..."
    )

    # --------------------------------------------------------
    # RUN NETWORK
    # --------------------------------------------------------
    #
    # SharedClockNetwork.run() stores the results internally
    # rather than returning a results dictionary.
    # --------------------------------------------------------

    network.run()

    # --------------------------------------------------------
    # Record trial number
    # --------------------------------------------------------

    trial_history.append(
        trial
    )

    # --------------------------------------------------------
    # Extract spike times
    # --------------------------------------------------------
    #
    # The network stores spike times in:
    #
    #     network.spike_history
    # --------------------------------------------------------

    n1_spike = network.spike_history[
        "N1"
    ][0]

    n2_spike = network.spike_history[
        "N2"
    ][0]

    n3_spike = network.spike_history[
        "N3"
    ][0]

    n4_spike = network.spike_history[
        "N4"
    ][0]

    # --------------------------------------------------------
    # Store spike times
    # --------------------------------------------------------

    spike_n1.append(
        n1_spike
    )

    spike_n2.append(
        n2_spike
    )

    spike_n3.append(
        n3_spike
    )

    spike_n4.append(
        n4_spike
    )

    # ========================================================
    # CALCULATE SPIKE-TIMING DIFFERENCES
    # ========================================================

    dt_n1_n3 = (
        n3_spike
        - n1_spike
    )

    dt_n2_n3 = (
        n3_spike
        - n2_spike
    )

    dt_n3_n4 = (
        n4_spike
        - n3_spike
    )

    # --------------------------------------------------------
    # Store Δt values
    # --------------------------------------------------------

    delta_t_n1_n3.append(
        dt_n1_n3
    )

    delta_t_n2_n3.append(
        dt_n2_n3
    )

    delta_t_n3_n4.append(
        dt_n3_n4
    )

    # ========================================================
    # CALCULATE TOTAL NETWORK PROPAGATION
    # ========================================================

    propagation = (
        n4_spike
        - n1_spike
    )

    propagation_n1_n4.append(
        propagation
    )

    # ========================================================
    # APPLY STDP
    # ========================================================

    result_n1_n3 = synapse_n1_n3.apply_stdp(
        n1_spike,
        n3_spike,
        STDP_PARAMETERS
    )

    result_n2_n3 = synapse_n2_n3.apply_stdp(
        n2_spike,
        n3_spike,
        STDP_PARAMETERS
    )

    result_n3_n4 = synapse_n3_n4.apply_stdp(
        n3_spike,
        n4_spike,
        STDP_PARAMETERS
    )

    # ========================================================
    # STORE UPDATED WEIGHTS
    # ========================================================

    weight_n1_n3.append(
        synapse_n1_n3.weight
    )

    weight_n2_n3.append(
        synapse_n2_n3.weight
    )

    weight_n3_n4.append(
        synapse_n3_n4.weight
    )

    # ========================================================
    # PRINT TRIAL SUMMARY
    # ========================================================

    print(
        f"  N1 spike: {n1_spike:.4f} ms"
    )

    print(
        f"  N2 spike: {n2_spike:.4f} ms"
    )

    print(
        f"  N3 spike: {n3_spike:.4f} ms"
    )

    print(
        f"  N4 spike: {n4_spike:.4f} ms"
    )

    print()

    print(
        f"  N1 → N3:"
        f" Δt={dt_n1_n3:.4f} ms"
        f"  weight={synapse_n1_n3.weight:.6f}"
    )

    print(
        f"  N2 → N3:"
        f" Δt={dt_n2_n3:.4f} ms"
        f"  weight={synapse_n2_n3.weight:.6f}"
    )

    print(
        f"  N3 → N4:"
        f" Δt={dt_n3_n4:.4f} ms"
        f"  weight={synapse_n3_n4.weight:.6f}"
    )

    print()

    print(
        f"  N1 → N4 propagation:"
        f" {propagation:.4f} ms"
    )

    print()


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL LEARNING RESULTS")
print("=" * 70)

print()

print("SYNAPTIC WEIGHTS")
print("-" * 70)

print(
    f"N1 → N3:"
    f" {weight_n1_n3[0]:.6f}"
    f" → {weight_n1_n3[-1]:.6f}"
    f"  "
    f"Δ={weight_n1_n3[-1] - weight_n1_n3[0]:+.6f}"
)

print(
    f"N2 → N3:"
    f" {weight_n2_n3[0]:.6f}"
    f" → {weight_n2_n3[-1]:.6f}"
    f"  "
    f"Δ={weight_n2_n3[-1] - weight_n2_n3[0]:+.6f}"
)

print(
    f"N3 → N4:"
    f" {weight_n3_n4[0]:.6f}"
    f" → {weight_n3_n4[-1]:.6f}"
    f"  "
    f"Δ={weight_n3_n4[-1] - weight_n3_n4[0]:+.6f}"
)

print()

print("SPIKE TIMING")
print("-" * 70)

print(
    f"N1:"
    f" {spike_n1[0]:.4f}"
    f" → {spike_n1[-1]:.4f} ms"
)

print(
    f"N2:"
    f" {spike_n2[0]:.4f}"
    f" → {spike_n2[-1]:.4f} ms"
)

print(
    f"N3:"
    f" {spike_n3[0]:.4f}"
    f" → {spike_n3[-1]:.4f} ms"
)

print(
    f"N4:"
    f" {spike_n4[0]:.4f}"
    f" → {spike_n4[-1]:.4f} ms"
)

print()

print("STDP TIMING")
print("-" * 70)

print(
    f"N1 → N3:"
    f" {delta_t_n1_n3[0]:.4f}"
    f" → {delta_t_n1_n3[-1]:.4f} ms"
)

print(
    f"N2 → N3:"
    f" {delta_t_n2_n3[0]:.4f}"
    f" → {delta_t_n2_n3[-1]:.4f} ms"
)

print(
    f"N3 → N4:"
    f" {delta_t_n3_n4[0]:.4f}"
    f" → {delta_t_n3_n4[-1]:.4f} ms"
)

print()

print("NETWORK PROPAGATION")
print("-" * 70)

print(
    f"N1 → N4:"
    f" {propagation_n1_n4[0]:.4f}"
    f" → {propagation_n1_n4[-1]:.4f} ms"
)

print()


# ============================================================
# PLOT 1
# SYNAPTIC WEIGHT VS TRIAL
# ============================================================

plt.figure()

plt.plot(
    trial_history,
    weight_n1_n3,
    marker="o",
    label="N1 → N3"
)

plt.plot(
    trial_history,
    weight_n2_n3,
    marker="o",
    label="N2 → N3"
)

plt.plot(
    trial_history,
    weight_n3_n4,
    marker="o",
    label="N3 → N4"
)

plt.xlabel(
    "Learning Trial"
)

plt.ylabel(
    "Synaptic Weight"
)

plt.title(
    "STDP Synaptic Weight Learning"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "stdp_weight_learning.png",
    dpi=300
)


# ============================================================
# PLOT 2
# SPIKE TIME VS TRIAL
# ============================================================

plt.figure()

plt.plot(
    trial_history,
    spike_n1,
    marker="o",
    label="N1"
)

plt.plot(
    trial_history,
    spike_n2,
    marker="o",
    label="N2"
)

plt.plot(
    trial_history,
    spike_n3,
    marker="o",
    label="N3"
)

plt.plot(
    trial_history,
    spike_n4,
    marker="o",
    label="N4"
)

plt.xlabel(
    "Learning Trial"
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

plt.savefig(
    "spike_time_learning.png",
    dpi=300
)


# ============================================================
# PLOT 3
# Δt VS TRIAL
# ============================================================

plt.figure()

plt.plot(
    trial_history,
    delta_t_n1_n3,
    marker="o",
    label="N1 → N3"
)

plt.plot(
    trial_history,
    delta_t_n2_n3,
    marker="o",
    label="N2 → N3"
)

plt.plot(
    trial_history,
    delta_t_n3_n4,
    marker="o",
    label="N3 → N4"
)

plt.xlabel(
    "Learning Trial"
)

plt.ylabel(
    "Δt (ms)"
)

plt.title(
    "Pre/Post-Synaptic Timing Difference"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "stdp_delta_t_learning.png",
    dpi=300
)


# ============================================================
# PLOT 4
# NETWORK PROPAGATION VS TRIAL
# ============================================================

plt.figure()

plt.plot(
    trial_history,
    propagation_n1_n4,
    marker="o"
)

plt.xlabel(
    "Learning Trial"
)

plt.ylabel(
    "Propagation Time (ms)"
)

plt.title(
    "Network Propagation Time Across Learning"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "network_propagation_learning.png",
    dpi=300
)


# ============================================================
# DISPLAY FIGURES
# ============================================================

print()
print("=" * 70)
print("LEARNING CURVES GENERATED")
print("=" * 70)

print()
print("Created:")

print(
    "  stdp_weight_learning.png"
)

print(
    "  spike_time_learning.png"
)

print(
    "  stdp_delta_t_learning.png"
)

print(
    "  network_propagation_learning.png"
)

print()

plt.show()