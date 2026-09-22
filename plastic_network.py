# ============================================================
# PLASTIC NEURAL NETWORK
#
# SIX-PHASE NEURONS
#        ↓
# SYNAPTIC COMMUNICATION
#        ↓
# DIVERGENCE + CONVERGENCE
#        ↓
# SPIKE TIMING
#        ↓
# STDP
#        ↓
# SYNAPTIC WEIGHT CHANGE
#        ↓
# FUTURE NETWORK BEHAVIOR
# ============================================================


from neuron import Neuron
from synapse import Synapse
from stdp_models import STDPParameters


class PlasticNetwork:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        simulation_time=50.0,
        dt=0.01,
        trials=20
    ):

        self.simulation_time = simulation_time
        self.dt = dt
        self.trials = trials

        # ----------------------------------------------------
        # STDP PARAMETERS
        # ----------------------------------------------------

        self.stdp_parameters = STDPParameters()

        # ----------------------------------------------------
        # NEURONS
        # ----------------------------------------------------

        self.neurons = {}

        # ----------------------------------------------------
        # SYNAPSES
        # ----------------------------------------------------

        self.synapses = []

        # ----------------------------------------------------
        # NETWORK HISTORY
        # ----------------------------------------------------

        self.trial_history = []

        self.weight_history = {}

        self.spike_history = {}

        # ----------------------------------------------------
        # BUILD NETWORK
        # ----------------------------------------------------

        self._build_network()


    # ========================================================
    # BUILD NETWORK
    #
    # Topology:
    #
    #
    #                 N2
    #                ↙  ↘
    #
    #              N1      N4 ──→ N5
    #                ↘  ↗
    #
    #                 N3
    #
    #
    # N1 diverges to N2 and N3.
    #
    # N2 and N3 converge onto N4.
    #
    # N4 then drives N5.
    # ========================================================

    def _build_network(self):

        # ----------------------------------------------------
        # CREATE NEURONS
        # ----------------------------------------------------

        for name in [
            "N1",
            "N2",
            "N3",
            "N4",
            "N5"
        ]:

            self.neurons[name] = Neuron(
                name=name
            )

        # ----------------------------------------------------
        # CREATE SYNAPSES
        # ----------------------------------------------------

        self._add_synapse(
            "N1",
            "N2",
            weight=5.0
        )

        self._add_synapse(
            "N1",
            "N3",
            weight=5.0
        )

        self._add_synapse(
            "N2",
            "N4",
            weight=5.0
        )

        self._add_synapse(
            "N3",
            "N4",
            weight=5.0
        )

        self._add_synapse(
            "N4",
            "N5",
            weight=5.0
        )


    # ========================================================
    # ADD SYNAPSE
    # ========================================================

    def _add_synapse(
        self,
        pre_name,
        post_name,
        weight
    ):

        synapse = Synapse(
            pre_neuron=self.neurons[pre_name],
            post_neuron=self.neurons[post_name],
            weight=weight
        )

        self.synapses.append(
            synapse
        )

        # ----------------------------------------------------
        # CREATE HISTORY FOR THIS CONNECTION
        # ----------------------------------------------------

        key = f"{pre_name}->{post_name}"

        self.weight_history[key] = []

        self.spike_history[key] = []


    # ========================================================
    # RESET NEURONS
    #
    # IMPORTANT:
    #
    # Neurons reset.
    #
    # Synapses DO NOT reset.
    #
    # Therefore learned weights survive between trials.
    # ========================================================

    def _reset_neurons(self):

        for neuron in self.neurons.values():

            neuron.reset_state()


    # ========================================================
    # FIND SOURCE NEURONS
    #
    # A source neuron has no incoming synapses.
    # ========================================================

    def _find_source_neurons(self):

        incoming = {
            name: 0
            for name in self.neurons
        }

        for synapse in self.synapses:

            post_name = (
                synapse.post_neuron.name
            )

            incoming[post_name] += 1

        return [
            name
            for name, count in incoming.items()
            if count == 0
        ]


    # ========================================================
    # BUILD OUTGOING CONNECTION MAP
    # ========================================================

    def _build_outgoing_map(self):

        outgoing = {
            name: []
            for name in self.neurons
        }

        for synapse in self.synapses:

            pre_name = (
                synapse.pre_neuron.name
            )

            outgoing[pre_name].append(
                synapse
            )

        return outgoing


    # ========================================================
    # RUN ONE TRIAL
    # ========================================================

    def run_trial(self):

        # ----------------------------------------------------
        # RESET NEURON STATES
        # ----------------------------------------------------

        self._reset_neurons()

        # ----------------------------------------------------
        # BUILD NETWORK MAP
        # ----------------------------------------------------

        outgoing = (
            self._build_outgoing_map()
        )

        source_neurons = (
            self._find_source_neurons()
        )

        # ----------------------------------------------------
        # EVENT QUEUE
        #
        # key = integer simulation step
        # value = list of synaptic events
        # ----------------------------------------------------

        pending_events = {}

        # ----------------------------------------------------
        # RECORD SPIKES
        # ----------------------------------------------------

        trial_spikes = {
            name: []
            for name in self.neurons
        }

        # ----------------------------------------------------
        # GLOBAL CLOCK
        # ----------------------------------------------------

        step_number = 0

        total_steps = int(
            self.simulation_time / self.dt
        ) + 1

        # ====================================================
        # GLOBAL SIMULATION LOOP
        # ====================================================

        while step_number <= total_steps:

            current_time = (
                step_number * self.dt
            )

            # ================================================
            # DELIVER EVENTS
            # ================================================

            events = pending_events.pop(
                step_number,
                []
            )

            for event in events:

                destination = (
                    event["synapse"].post_neuron
                )

                destination.synaptic_events.append(
                    (
                        current_time,
                        event["signal"]
                    )
                )

                destination.externally_driven = True

            # ================================================
            # STEP EVERY NEURON
            # ================================================

            newly_spiking = []

            for name, neuron in (
                self.neurons.items()
            ):

                previous_count = len(
                    neuron.spike_times
                )

                # --------------------------------------------
                # SOURCE NEURONS CAN TRIGGER INTRINSICALLY.
                #
                # DOWNSTREAM NEURONS REQUIRE SYNAPTIC INPUT.
                # --------------------------------------------

                allow_intrinsic = (
                    name in source_neurons
                )

                neuron.step(
                    allow_intrinsic_trigger=
                    allow_intrinsic
                )

                # --------------------------------------------
                # CHECK FOR NEW SPIKE
                # --------------------------------------------

                if len(
                    neuron.spike_times
                ) > previous_count:

                    new_spikes = (
                        neuron.spike_times[
                            previous_count:
                        ]
                    )

                    for spike_time in new_spikes:

                        trial_spikes[
                            name
                        ].append(
                            spike_time
                        )

                        newly_spiking.append(
                            (
                                name,
                                spike_time
                            )
                        )

            # ================================================
            # TRANSMIT NEW SPIKES
            # ================================================

            for (
                source_name,
                spike_time
            ) in newly_spiking:

                for synapse in outgoing[
                    source_name
                ]:

                    delivery_step = (
                        step_number + 1
                    )

                    event = {
                        "synapse": synapse,
                        "signal": synapse.weight
                    }

                    if delivery_step not in (
                        pending_events
                    ):

                        pending_events[
                            delivery_step
                        ] = []

                    pending_events[
                        delivery_step
                    ].append(
                        event
                    )

            # ================================================
            # ADVANCE GLOBAL CLOCK
            # ================================================

            step_number += 1

        return trial_spikes


    # ========================================================
    # APPLY STDP
    #
    # Every synapse learns independently.
    # ========================================================

    def apply_stdp(
        self,
        trial_spikes
    ):

        results = []

        for synapse in self.synapses:

            pre_name = (
                synapse.pre_neuron.name
            )

            post_name = (
                synapse.post_neuron.name
            )

            pre_spikes = (
                trial_spikes[pre_name]
            )

            post_spikes = (
                trial_spikes[post_name]
            )

            # ------------------------------------------------
            # A SYNAPSE CANNOT LEARN WITHOUT BOTH SPIKES.
            # ------------------------------------------------

            if (
                not pre_spikes
                or not post_spikes
            ):

                results.append({
                    "connection":
                        f"{pre_name}->{post_name}",
                    "updated": False
                })

                continue

            # ------------------------------------------------
            # FIRST PRE/POST SPIKE PAIR
            # ------------------------------------------------

            pre_spike = pre_spikes[0]

            post_spike = post_spikes[0]

            # ------------------------------------------------
            # APPLY EXISTING STDP RULE
            # ------------------------------------------------

            result = synapse.apply_stdp(
                pre_spike_time=pre_spike,
                post_spike_time=post_spike,
                parameters=self.stdp_parameters
            )

            connection = (
                f"{pre_name}->{post_name}"
            )

            # ------------------------------------------------
            # RECORD HISTORY
            # ------------------------------------------------

            self.weight_history[
                connection
            ].append(
                synapse.weight
            )

            self.spike_history[
                connection
            ].append({
                "pre": pre_spike,
                "post": post_spike,
                "delta_t": result.delta_t,
                "delta_w": result.delta_w
            })

            results.append({
                "connection": connection,
                "updated": True,
                "delta_t": result.delta_t,
                "delta_w": result.delta_w,
                "old_weight": result.old_weight,
                "new_weight": result.new_weight
            })

        return results


    # ========================================================
    # RUN COMPLETE LEARNING EXPERIMENT
    # ========================================================

    def run(self):

        print()
        print("======================================")
        print("PLASTIC NEURAL NETWORK")
        print("======================================")

        print()
        print("NETWORK TOPOLOGY")
        print()
        print("N1 → N2 → N4 → N5")
        print("N1 → N3 → N4")
        print()

        # ====================================================
        # LEARNING TRIALS
        # ====================================================

        for trial in range(
            1,
            self.trials + 1
        ):

            # ------------------------------------------------
            # RUN NETWORK
            # ------------------------------------------------

            trial_spikes = (
                self.run_trial()
            )

            # ------------------------------------------------
            # APPLY STDP
            # ------------------------------------------------

            stdp_results = (
                self.apply_stdp(
                    trial_spikes
                )
            )

            # ------------------------------------------------
            # SAVE TRIAL
            # ------------------------------------------------

            self.trial_history.append({
                "trial": trial,
                "spikes": trial_spikes,
                "stdp": stdp_results
            })

            # =================================================
            # PRINT TRIAL
            # =================================================

            print(
                f"\nTRIAL {trial:02d}"
            )

            print(
                "------------------------------"
            )

            for name in self.neurons:

                print(
                    f"{name}: "
                    f"{trial_spikes[name]}"
                )

            print()

            for result in stdp_results:

                if result["updated"]:

                    print(
                        f"{result['connection']} | "
                        f"Δt = "
                        f"{result['delta_t']:.4f} ms | "
                        f"Δw = "
                        f"{result['delta_w']:.6f} | "
                        f"weight = "
                        f"{result['new_weight']:.6f}"
                    )

                else:

                    print(
                        f"{result['connection']} | "
                        f"NO UPDATE"
                    )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        self.print_final_results()


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    def print_final_results(self):

        print()
        print("======================================")
        print("FINAL NETWORK STATE")
        print("======================================")

        print()

        for synapse in self.synapses:

            pre_name = (
                synapse.pre_neuron.name
            )

            post_name = (
                synapse.post_neuron.name
            )

            connection = (
                f"{pre_name}->{post_name}"
            )

            history = (
                self.weight_history[
                    connection
                ]
            )

            print(
                f"{connection}"
            )

            print(
                f"  Initial weight: "
                f"{history[0] if history else synapse.weight:.6f}"
            )

            print(
                f"  Final weight:   "
                f"{synapse.weight:.6f}"
            )

            if self.spike_history[
                connection
            ]:

                first = self.spike_history[
                    connection
                ][0]

                last = self.spike_history[
                    connection
                ][-1]

                print(
                    f"  Initial Δt:     "
                    f"{first['delta_t']:.4f} ms"
                )

                print(
                    f"  Final Δt:       "
                    f"{last['delta_t']:.4f} ms"
                )

            print()


# ============================================================
# RUN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    network = PlasticNetwork(
        simulation_time=50.0,
        dt=0.01,
        trials=20
    )

    network.run()