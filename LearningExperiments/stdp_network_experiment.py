# ==========================================
# STDP NETWORK EXPERIMENT
#
# CONNECTS:
#
#   SIX-PHASE NEURON
#          ↓
#   SYNAPTIC COMMUNICATION
#          ↓
#   SHARED-CLOCK NETWORK
#          ↓
#   STDP PLASTICITY
#
# The actual Synapse.weight is modified
# after each trial.
# ==========================================

from neuron import Neuron
from synapse import Synapse
from stdp_models import STDPParameters


class STDPNetworkExperiment:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        simulation_time=50.0,
        dt=0.01,
        trials=20
    ):

        self.simulation_time = simulation_time
        self.dt = dt
        self.trials = trials

        # --------------------------------------
        # STDP PARAMETERS
        # --------------------------------------

        self.stdp_parameters = STDPParameters()

        # --------------------------------------
        # LEARNING HISTORY
        # --------------------------------------

        self.weight_history = []

        self.delta_t_history = []

        self.pre_spike_history = []

        self.post_spike_history = []

        self.delta_w_history = []


    # ==========================================
    # RUN ONE NETWORK TRIAL
    # ==========================================

    def run_trial(
        self,
        synapse
    ):

        # --------------------------------------
        # CREATE FRESH NEURONS
        # --------------------------------------

        pre_neuron = Neuron(
            name="N1"
        )

        post_neuron = Neuron(
            name="N2"
        )

        # --------------------------------------
        # CREATE NETWORK STATE
        # --------------------------------------

        current_time = 0.0

        pending_events = {}

        pre_spikes = []

        post_spikes = []

        # --------------------------------------
        # RECORD INITIAL WEIGHT
        # --------------------------------------

        weight = synapse.weight

        # ======================================
        # SIMULATION LOOP
        # ======================================

        while current_time <= self.simulation_time:

            # ==================================
            # DELIVER SYNAPTIC EVENTS
            # ==================================

            events = pending_events.pop(
                current_time,
                []
            )

            for signal in events:

                post_neuron.synaptic_events.append(
                    (
                        current_time,
                        signal
                    )
                )

                post_neuron.externally_driven = True

            # ==================================
            # STEP PRE-SYNAPTIC NEURON
            # ==================================

            previous_pre_count = len(
                pre_neuron.spike_times
            )

            pre_neuron.step(
                allow_intrinsic_trigger=True
            )

            # ==================================
            # DETECT PRE SPIKE
            # ==================================

            if (
                len(pre_neuron.spike_times)
                >
                previous_pre_count
            ):

                new_pre_spikes = (
                    pre_neuron.spike_times[
                        previous_pre_count:
                    ]
                )

                for spike_time in new_pre_spikes:

                    pre_spikes.append(
                        spike_time
                    )

                    # --------------------------
                    # TRANSMIT THROUGH SYNAPSE
                    # --------------------------

                    delivery_time = (
                        current_time
                        + self.dt
                    )

                    if (
                        delivery_time
                        not in pending_events
                    ):

                        pending_events[
                            delivery_time
                        ] = []

                    pending_events[
                        delivery_time
                    ].append(
                        weight
                    )

            # ==================================
            # STEP POST-SYNAPTIC NEURON
            # ==================================

            previous_post_count = len(
                post_neuron.spike_times
            )

            post_neuron.step(
                allow_intrinsic_trigger=False
            )

            # ==================================
            # DETECT POST SPIKE
            # ==================================

            if (
                len(post_neuron.spike_times)
                >
                previous_post_count
            ):

                new_post_spikes = (
                    post_neuron.spike_times[
                        previous_post_count:
                    ]
                )

                for spike_time in new_post_spikes:

                    post_spikes.append(
                        spike_time
                    )

            # ==================================
            # ADVANCE GLOBAL CLOCK
            # ==================================

            current_time += self.dt

        # ======================================
        # VERIFY SPIKES
        # ======================================

        if len(pre_spikes) == 0:

            raise RuntimeError(
                "Presynaptic neuron did not spike."
            )

        if len(post_spikes) == 0:

            raise RuntimeError(
                "Postsynaptic neuron did not spike."
            )

        # ======================================
        # FIRST SPIKE PAIR
        # ======================================

        pre_spike = pre_spikes[0]

        post_spike = post_spikes[0]

        # ======================================
        # APPLY STDP TO ACTUAL SYNAPSE
        # ======================================

        result = synapse.apply_stdp(
            pre_spike_time=pre_spike,
            post_spike_time=post_spike,
            parameters=self.stdp_parameters
        )

        return result


    # ==========================================
    # RUN LEARNING EXPERIMENT
    # ==========================================

    def run(self):

        print()
        print("======================================")
        print("STDP NETWORK LEARNING EXPERIMENT")
        print("======================================")

        # ======================================
        # CREATE PERSISTENT SYNAPSE
        #
        # This is critical.
        #
        # The same synapse survives across
        # trials so learning accumulates.
        # ======================================

        pre_neuron = Neuron(
            name="N1"
        )

        post_neuron = Neuron(
            name="N2"
        )

        synapse = Synapse(
            pre_neuron=pre_neuron,
            post_neuron=post_neuron,
            weight=5.0
        )

        # ======================================
        # LEARNING TRIALS
        # ======================================

        for trial in range(
            1,
            self.trials + 1
        ):

            # ----------------------------------
            # STORE WEIGHT BEFORE TRIAL
            # ----------------------------------

            weight_before = synapse.weight

            # ----------------------------------
            # RUN NETWORK TRIAL
            # ----------------------------------

            result = self.run_trial(
                synapse
            )

            # ----------------------------------
            # STORE LEARNING HISTORY
            # ----------------------------------

            self.weight_history.append(
                synapse.weight
            )

            self.delta_t_history.append(
                result.delta_t
            )

            self.pre_spike_history.append(
                result.pre_spike_time
            )

            self.post_spike_history.append(
                result.post_spike_time
            )

            self.delta_w_history.append(
                result.delta_w
            )

            # ----------------------------------
            # PRINT TRIAL
            # ----------------------------------

            print(
                f"Trial {trial:02d} | "
                f"Δt = "
                f"{result.delta_t:.4f} ms | "
                f"Δw = "
                f"{result.delta_w:.6f} | "
                f"weight = "
                f"{weight_before:.6f} → "
                f"{synapse.weight:.6f}"
            )

        # ======================================
        # FINAL RESULTS
        # ======================================

        print()
        print("======================================")
        print("LEARNING COMPLETE")
        print("======================================")

        print()
        print(
            "Initial weight:"
        )

        print(
            5.0
        )

        print()
        print(
            "Final weight:"
        )

        print(
            synapse.weight
        )

        print()
        print(
            "Initial Δt:"
        )

        print(
            self.delta_t_history[0]
        )

        print()
        print(
            "Final Δt:"
        )

        print(
            self.delta_t_history[-1]
        )


# ==========================================
# DIRECT TEST
# ==========================================

if __name__ == "__main__":

    experiment = STDPNetworkExperiment(
        simulation_time=50.0,
        dt=0.01,
        trials=20
    )

    experiment.run()