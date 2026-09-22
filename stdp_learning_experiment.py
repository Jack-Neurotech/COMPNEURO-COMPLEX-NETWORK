# ==========================================
# STDP LEARNING EXPERIMENT
# VALIDATING THE EXISTING STDP MODEL
# ==========================================

from synapse import Synapse


# ==========================================
# CREATE A SYNAPSE
# ==========================================

synapse = Synapse(
    pre_neuron="Neuron 1",
    post_neuron="Neuron 2",
    weight=5.0
)


# ==========================================
# EXPERIMENT CASES
# ==========================================

experiments = [

    {
        "name": "Pre before Post - close",
        "pre": 20.0,
        "post": 21.0
    },

    {
        "name": "Pre before Post - farther apart",
        "pre": 20.0,
        "post": 25.0
    },

    {
        "name": "Post before Pre",
        "pre": 20.0,
        "post": 19.0
    },

    {
        "name": "Exactly simultaneous",
        "pre": 20.0,
        "post": 20.0
    }

]


# ==========================================
# RUN EXPERIMENTS
# ==========================================

print("======================================")
print("STDP LEARNING EXPERIMENT")
print("======================================")


for experiment in experiments:

    print()
    print("--------------------------------------")
    print(experiment["name"])
    print("--------------------------------------")


    # --------------------------------------
    # RESET WEIGHT FOR EACH EXPERIMENT
    # --------------------------------------

    synapse.weight = 5.0


    # --------------------------------------
    # GET SPIKE TIMES
    # --------------------------------------

    pre_spike = experiment["pre"]

    post_spike = experiment["post"]


    # --------------------------------------
    # APPLY STDP
    # --------------------------------------

    result = synapse.apply_stdp(
        pre_spike_time=pre_spike,
        post_spike_time=post_spike
    )


    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    print("Pre-spike:")
    print(pre_spike, "ms")

    print("Post-spike:")
    print(post_spike, "ms")

    print("Delta t:")
    print(result["delta_t"], "ms")

    print("Delta w:")
    print(result["delta_w"])

    print("Initial weight:")
    print(5.0)

    print("New weight:")
    print(result["new_weight"])