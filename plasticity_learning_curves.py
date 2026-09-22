# ============================================================
# STDP PLASTICITY LEARNING CURVES
# ============================================================
#
# Measures how repeated STDP changes:
#
#   1. Synaptic weights
#   2. Neuron spike times
#   3. Pre/post spike timing
#   4. N1 -> N4 propagation time
#
# Uses the EXISTING SharedClockNetwork architecture.
#
# No changes are made to:
#
#   neuron.py
#   synapse.py
#   shared_clock_network.py
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from neuron import Neuron
from shared_clock_network import SharedClockNetwork
from stdp_models import STDPParameters


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

TRIALS = 20

INITIAL_WEIGHT = 5.0

SIMULATION_TIME = 50.0

DT = 0.01


# ============================================================
# STDP PARAMETERS
# ============================================================

stdp_parameters = STDPParameters()


# ============================================================
# SYNAPTIC CONNECTION DEFINITIONS
# ============================================================

CONNECTION_NAMES = [
    "N1->N3",
    "N2->N3",
    "N3->N4",
]


# ============================================================
# CURRENT WEIGHTS
# ============================================================

weights = {
    "N1->N3": INITIAL_WEIGHT,
    "N2->N3": INITIAL_WEIGHT,
    "N3->N4": INITIAL_WEIGHT,
}


# ============================================================
# EXPERIMENT HISTORY
# ============================================================

weight_history = {
    name: []
    for name in CONNECTION_NAMES
}


delta_t_history = {
    name: []
    for name in CONNECTION_NAMES
}


spike_history = {
    "N1": [],
    "N2": [],
    "N3": [],
    "N4": [],
}


propagation_history = []


# ============================================================
# CREATE NETWORK
# ============================================================

def create_network(current_weights):

    # --------------------------------------------------------
    # Create neurons
    # --------------------------------------------------------

    N1 = Neuron("N1")
    N2 = Neuron("N2")
    N3 = Neuron("N3")
    N4 = Neuron("N4")

    # --------------------------------------------------------
    # Create shared-clock network
    # --------------------------------------------------------

    network = SharedClockNetwork(
        simulation_time=SIMULATION_TIME,
        dt=DT
    )

    # --------------------------------------------------------
    # Add neurons
    # --------------------------------------------------------

    network.add_neuron(N1)
    network.add_neuron(N2)
    network.add_neuron(N3)
    network.add_neuron(N4)

    # --------------------------------------------------------
    # Connect N1 -> N3
    # --------------------------------------------------------

    s13 = network.connect(
        N1,
        N3,
        weight=current_weights["N1->N3"]
    )

    # --------------------------------------------------------
    # Connect N2 -> N3
    # --------------------------------------------------------

    s23 = network.connect(
        N2,
        N3,
        weight=current_weights["N2->N3"]
    )

    # --------------------------------------------------------
    # Connect N3 -> N4
    # --------------------------------------------------------

    s34 = network.connect(
        N3,
        N4,
        weight=current_weights["N3->N4"]
    )

    return network, {
        "N1->N3": s13,
        "N2->N3": s23,
        "N3->N4": s34,
    }


# ============================================================
# RUN LEARNING EXPERIMENT
# ============================================================

for trial in range(1, TRIALS + 1):

    print()
    print("=" * 70)
    print(f"TRIAL {trial:02d}")
    print("=" * 70)

    # --------------------------------------------------------
    # Build fresh neurons/network.
    #
    # Synaptic weights come from the previous trial.
    # --------------------------------------------------------

    network, synapses = create_network(weights)

    # --------------------------------------------------------
    # Run network
    # --------------------------------------------------------

    network.run()

    # --------------------------------------------------------
    # Record spike times
    # --------------------------------------------------------

    for neuron_name in spike_history:

        spikes = network.spike_history[neuron_name]

        if spikes:

            spike_history[neuron_name].append(
                spikes[0]
            )

        else:

            spike_history[neuron_name].append(
                None
            )

    # --------------------------------------------------------
    # Apply STDP
    # --------------------------------------------------------

    for connection_name, synapse in synapses.items():

        pre_neuron = synapse.pre_neuron
        post_neuron = synapse.post_neuron

        if (
            pre_neuron.spike_times
            and
            post_neuron.spike_times
        ):

            pre_time = pre_neuron.spike_times[0]

            post_time = post_neuron.spike_times[0]

            result = synapse.apply_stdp(
                pre_time,
                post_time,
                stdp_parameters
            )

            delta_t_history[
                connection_name
            ].append(
                result.delta_t
            )

        else:

            delta_t_history[
                connection_name
            ].append(
                None
            )

    # --------------------------------------------------------
    # Store updated weights
    # --------------------------------------------------------

    for connection_name, synapse in synapses.items():

        weights[connection_name] = synapse.weight

        weight_history[
            connection_name
        ].append(
            synapse.weight
        )

    # --------------------------------------------------------
    # Calculate propagation time
    #
    # N1 -> N3 -> N4
    # --------------------------------------------------------

    N1_spike = spike_history["N1"][-1]

    N4_spike = spike_history["N4"][-1]

    if (
        N1_spike is not None
        and
        N4_spike is not None
    ):

        propagation_time = (
            N4_spike
            -
            N1_spike
        )

        propagation_history.append(
            propagation_time
        )

    else:

        propagation_history.append(None)

    # --------------------------------------------------------
    # Print trial summary
    # --------------------------------------------------------

    print()

    for neuron_name in spike_history:

        spike = spike_history[neuron_name][-1]

        print(
            f"{neuron_name}: "
            f"{spike:.4f} ms"
        )

    print()

    for connection_name in CONNECTION_NAMES:

        delta_t = delta_t_history[
            connection_name
        ][-1]

        weight = weight_history[
            connection_name
        ][-1]

        print(
            f"{connection_name} | "
            f"Δt = {delta_t:.4f} ms | "
            f"weight = {weight:.6f}"
        )


# ============================================================
# FINAL EXPERIMENT SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL LEARNING SUMMARY")
print("=" * 70)


# ============================================================
# WEIGHT LEARNING
# ============================================================

print()
print("SYNAPTIC WEIGHTS")
print("-" * 70)

for connection_name in CONNECTION_NAMES:

    initial = INITIAL_WEIGHT

    final = weight_history[
        connection_name
    ][-1]

    change = final - initial

    print(
        f"{connection_name}: "
        f"{initial:.6f} → "
        f"{final:.6f} "
        f"({change:+.6f})"
    )


# ============================================================
# SPIKE-TIMING CHANGES
# ============================================================

print()
print("SPIKE TIMING")
print("-" * 70)

for neuron_name in spike_history:

    initial = spike_history[
        neuron_name
    ][0]

    final = spike_history[
        neuron_name
    ][-1]

    change = final - initial

    print(
        f"{neuron_name}: "
        f"{initial:.4f} ms → "
        f"{final:.4f} ms "
        f"({change:+.4f} ms)"
    )


# ============================================================
# DELTA-T CHANGES
# ============================================================

print()
print("PRE/POST SPIKE TIMING")
print("-" * 70)

for connection_name in CONNECTION_NAMES:

    initial = delta_t_history[
        connection_name
    ][0]

    final = delta_t_history[
        connection_name
    ][-1]

    change = final - initial

    print(
        f"{connection_name}: "
        f"{initial:.4f} ms → "
        f"{final:.4f} ms "
        f"({change:+.4f} ms)"
    )


# ============================================================
# NETWORK PROPAGATION
# ============================================================

print()
print("NETWORK PROPAGATION")
print("-" * 70)

initial_propagation = propagation_history[0]

final_propagation = propagation_history[-1]

propagation_change = (
    final_propagation
    -
    initial_propagation
)

print(
    f"Initial N1 → N4: "
    f"{initial_propagation:.4f} ms"
)

print(
    f"Final N1 → N4:   "
    f"{final_propagation:.4f} ms"
)

print(
    f"Change: "
    f"{propagation_change:+.4f} ms"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)