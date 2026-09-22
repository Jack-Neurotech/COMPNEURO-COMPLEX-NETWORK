# ==========================================
# REPEATED STDP LEARNING EXPERIMENT
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
# EXPERIMENT PARAMETERS
# ==========================================

NUMBER_OF_TRIALS = 10


# ==========================================
# CREATE SYNAPSE
# ==========================================

synapse = Synapse(
    pre_neuron="Neuron 1",
    post_neuron="Neuron 2",
    weight=5.0
)


# ==========================================
# STORE LEARNING HISTORY
# ==========================================

learning_history = []


# ==========================================
# REPEATED LEARNING LOOP
# ==========================================

for trial in range(
    1,
    NUMBER_OF_TRIALS + 1
):

    print()
    print("======================================")
    print(f"TRIAL {trial}")
    print("======================================")


    # ======================================
    # STEP 1
    # GET NEURON 1 SPIKE
    # ======================================

    neuron_1_spikes = detect_spikes(
        times,
        voltages,
        threshold=V_threshold
    )


    if not neuron_1_spikes:

        print("Neuron 1 did not spike.")

        continue


    # Use the first presynaptic spike.

    pre_spike_time = neuron_1_spikes[0]


    print("N1 spike:")
    print(pre_spike_time, "ms")


    # ======================================
    # STEP 2
    # TRANSMIT THROUGH CURRENT SYNAPSE
    # ======================================

    transmission = synapse.transmit(
        pre_spike_time
    )


    synaptic_events = [
        (
            transmission["spike_time"],
            transmission["signal"]
        )
    ]


    print("Synaptic weight before trial:")
    print(synapse.weight)


    # ======================================
    # STEP 3
    # SIMULATE NEURON 2
    # ======================================

    neuron_2_times, neuron_2_voltages = (
        simulate_neuron(
            synaptic_events
        )
    )


    # ======================================
    # STEP 4
    # DETECT N2 SPIKE
    # ======================================

    neuron_2_spikes = detect_spikes(
        neuron_2_times,
        neuron_2_voltages,
        threshold=V_threshold
    )


    if not neuron_2_spikes:

        print("N2 did not spike.")

        learning_history.append(
            {
                "trial": trial,
                "pre_spike": pre_spike_time,
                "post_spike": None,
                "delta_t": None,
                "delta_w": None,
                "weight": synapse.weight
            }
        )

        continue


    # Use the first postsynaptic spike.

    post_spike_time = neuron_2_spikes[0]


    print("N2 spike:")
    print(post_spike_time, "ms")


    # ======================================
    # STEP 5
    # APPLY STDP
    # ======================================

    stdp_result = synapse.apply_stdp(
        pre_spike_time=pre_spike_time,
        post_spike_time=post_spike_time
    )


    # ======================================
    # STEP 6
    # STORE RESULTS
    # ======================================

    learning_history.append(
        {
            "trial": trial,
            "pre_spike": pre_spike_time,
            "post_spike": post_spike_time,
            "delta_t": stdp_result["delta_t"],
            "delta_w": stdp_result["delta_w"],
            "weight": stdp_result["new_weight"]
        }
    )


    # ======================================
    # DISPLAY STDP RESULT
    # ======================================

    print("Delta t:")
    print(
        stdp_result["delta_t"],
        "ms"
    )

    print("Delta w:")
    print(
        stdp_result["delta_w"]
    )

    print("Synaptic weight after trial:")
    print(
        stdp_result["new_weight"]
    )


# ==========================================
# FINAL LEARNING SUMMARY
# ==========================================

print()
print()
print("======================================")
print("LEARNING SUMMARY")
print("======================================")


for result in learning_history:

    print(
        f"Trial {result['trial']}: "
        f"weight = {result['weight']}"
    )