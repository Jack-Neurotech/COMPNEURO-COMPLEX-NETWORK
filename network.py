
# =========================================
# NETWORK
# MULTI-HOP EVENT PROPAGATION
#
# The individual Neuron class remains
# responsible for its own six-phase
# action potential.
#
# Synapse remains responsible for:
#
#   - transmission
#   - synaptic weight
#   - STDP
#
# This Network class is responsible for:
#
#   1. Holding neurons
#   2. Holding synapses
#   3. Finding source neurons
#   4. Running neurons
#   5. Detecting spike events
#   6. Propagating events through
#      multiple network layers
#
# Example:
#
#   N1 → N2 → N3
#
# N1 fires
#      ↓
# N2 receives N1
#      ↓
# N2 fires
#      ↓
# N3 receives N2
#
# ==========================================

from neuron import Neuron
from synapse import Synapse


# ==========================================
# NETWORK
# ==========================================

class Network:

    def __init__(
        self,
        simulation_time=50.0,
        dt=0.01
    ):

        self.simulation_time = simulation_time
        self.dt = dt

        self.neurons = {}
        self.synapses = []


    # ==========================================
    # ADD NEURON
    # ==========================================

    def add_neuron(
        self,
        neuron
    ):

        self.neurons[
            neuron.name
        ] = neuron


    # ==========================================
    # CONNECT NEURONS
    # ==========================================

    def connect(
        self,
        pre_neuron,
        post_neuron,
        weight=5.0
    ):

        synapse = Synapse(
            pre_neuron=pre_neuron,
            post_neuron=post_neuron,
            weight=weight
        )

        self.synapses.append(
            synapse
        )

        return synapse


    # ==========================================
    # FIND SOURCE NEURONS
    #
    # A source neuron has no incoming
    # synapses.
    # ==========================================

    def _find_source_neurons(self):

        source_neurons = []

        for neuron in self.neurons.values():

            has_incoming_connection = False

            for synapse in self.synapses:

                if (
                    synapse.post_neuron
                    == neuron
                ):

                    has_incoming_connection = True

                    break

            if not has_incoming_connection:

                source_neurons.append(
                    neuron
                )

        return source_neurons


    # ==========================================
    # BUILD OUTGOING CONNECTION MAP
    #
    # Example:
    #
    # N1 → [N2]
    # N2 → [N3]
    #
    # ==========================================

    def _build_outgoing_map(self):

        outgoing = {}

        for neuron in self.neurons.values():

            outgoing[neuron] = []

        for synapse in self.synapses:

            outgoing[
                synapse.pre_neuron
            ].append(
                synapse
            )

        return outgoing


    # ==========================================
    # RUN NETWORK
    #
    # Event-driven multi-hop propagation.
    # ==========================================

    def run(self):

        print()
        print("======================================")
        print("RUNNING MULTI-HOP NETWORK")
        print("======================================")


        # ======================================
        # FIND SOURCE NEURONS
        # ======================================

        source_neurons = (
            self._find_source_neurons()
        )


        # ======================================
        # BUILD CONNECTION MAP
        # ======================================

        outgoing = (
            self._build_outgoing_map()
        )


        # ======================================
        # SIMULATE SOURCE NEURONS
        # ======================================

        for neuron in source_neurons:

            neuron.simulate()


        # ======================================
        # QUEUE INITIAL SPIKE EVENTS
        # ======================================

        event_queue = []

        for neuron in source_neurons:

            for spike_time in (
                neuron.spike_times
            ):

                event_queue.append(
                    (
                        spike_time,
                        neuron
                    )
                )


        # ======================================
        # PROCESS EVENTS
        #
        # Every time a neuron fires:
        #
        #   1. Find outgoing synapses
        #   2. Transmit the spike
        #   3. Give the event to the
        #      downstream neuron
        #   4. Simulate downstream neuron
        #   5. Add its spike to the queue
        #
        # ======================================

        processed_events = set()


        while event_queue:

            spike_time, source = (
                event_queue.pop(0)
            )


            # ==================================
            # PREVENT DUPLICATE EVENT PROCESSING
            # ==================================

            event_key = (
                source,
                spike_time
            )

            if event_key in processed_events:

                continue

            processed_events.add(
                event_key
            )


            # ==================================
            # FIND OUTGOING SYNAPSES
            # ==================================

            outgoing_synapses = outgoing.get(
                source,
                []
            )


            for synapse in outgoing_synapses:


                # ==================================
                # TRANSMIT SPIKE
                # ==================================

                transmission = (
                    synapse.transmit(
                        spike_time
                    )
                )


                # ==================================
                # DOWNSTREAM NEURON
                # ==================================

                destination = (
                    synapse.post_neuron
                )


                # ==================================
                # SIMULATE DOWNSTREAM NEURON
                #
                # The transmitted spike becomes
                # a synaptic event for the next
                # neuron.
                # ==================================

                destination.simulate(
                    synaptic_events=[
                        (
                            transmission[
                                "spike_time"
                            ],
                            transmission[
                                "signal"
                            ]
                        )
                    ]
                )


                # ==================================
                # QUEUE NEW SPIKES
                #
                # These spikes can now propagate
                # into another network layer.
                # ==================================

                for downstream_spike in (
                    destination.spike_times
                ):

                    event_queue.append(
                        (
                            downstream_spike,
                            destination
                        )
                    )


        # ======================================
        # RETURN NETWORK
        # ======================================

        return self.neurons


# ==========================================
# DIRECT TEST
# ==========================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("NETWORK TEST")
    print("======================================")


    # ======================================
    # CREATE NEURONS
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
    # CREATE NETWORK
    # ======================================

    network = Network()

    network.add_neuron(N1)
    network.add_neuron(N2)
    network.add_neuron(N3)


    # ======================================
    # CREATE CHAIN
    # ======================================

    network.connect(
        pre_neuron=N1,
        post_neuron=N2,
        weight=5.0
    )

    network.connect(
        pre_neuron=N2,
        post_neuron=N3,
        weight=5.0
    )


    # ======================================
    # RUN
    # ======================================

    network.run()


    # ======================================
    # OUTPUT
    # ======================================

    print()

    print("N1 spikes:")
    print(N1.spike_times)

    print()

    print("N2 spikes:")
    print(N2.spike_times)

    print()

    print("N3 spikes:")
    print(N3.spike_times)
