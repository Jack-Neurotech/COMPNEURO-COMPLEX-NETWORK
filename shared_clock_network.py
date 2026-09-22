# ==========================================
# SHARED-CLOCK CONVERGENT NEURAL NETWORK
#
# SIX-PHASE NEURONS
# SYNAPTIC COMMUNICATION
# GLOBAL SHARED CLOCK
#
# TOPOLOGY:
#
#        ┌──────→ N3 ──────→ N4
#        │
# N1 ────┤
#        │
# N2 ────┘
#
# N1 and N2 are independent source neurons.
#
# N3 receives convergent input from BOTH.
#
# N4 receives input from N3.
# ==========================================

from neuron import Neuron
from synapse import Synapse


class SharedClockNetwork:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        simulation_time=50.0,
        dt=0.01
    ):

        self.simulation_time = simulation_time
        self.dt = dt

        # --------------------------------------
        # NETWORK OBJECTS
        # --------------------------------------

        self.neurons = {}
        self.synapses = []

        # --------------------------------------
        # NETWORK HISTORY
        # --------------------------------------

        self.spike_events = []

        self.voltage_history = {}

        self.spike_history = {}

        self.time_history = []

        # --------------------------------------
        # PENDING SYNAPTIC EVENTS
        # --------------------------------------

        self.pending_events = {}


    # ==========================================
    # ADD NEURON
    # ==========================================

    def add_neuron(
        self,
        neuron
    ):

        if neuron.name in self.neurons:

            raise ValueError(
                f"Neuron '{neuron.name}' already exists."
            )

        self.neurons[
            neuron.name
        ] = neuron


    # ==========================================
    # CONNECT TWO NEURONS
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
    # BUILD OUTGOING SYNAPSE MAP
    # ==========================================

    def _build_outgoing_map(self):

        outgoing = {}

        # --------------------------------------
        # INITIALIZE EVERY NEURON
        # --------------------------------------

        for neuron in self.neurons.values():

            outgoing[
                neuron
            ] = []

        # --------------------------------------
        # ASSIGN SYNAPSES
        # --------------------------------------

        for synapse in self.synapses:

            outgoing[
                synapse.pre_neuron
            ].append(
                synapse
            )

        return outgoing


    # ==========================================
    # FIND SOURCE NEURONS
    #
    # Source = zero incoming synapses.
    # ==========================================

    def _find_source_neurons(self):

        incoming = {
            neuron: 0
            for neuron in self.neurons.values()
        }

        # --------------------------------------
        # COUNT INCOMING CONNECTIONS
        # --------------------------------------

        for synapse in self.synapses:

            incoming[
                synapse.post_neuron
            ] += 1

        # --------------------------------------
        # RETURN SOURCE NEURONS
        # --------------------------------------

        return [
            neuron
            for neuron in self.neurons.values()
            if incoming[neuron] == 0
        ]


    # ==========================================
    # RESET NETWORK
    # ==========================================

    def reset(self):

        # --------------------------------------
        # RESET NEURONS
        # --------------------------------------

        for neuron in self.neurons.values():

            neuron.reset_state()

        # --------------------------------------
        # RESET HISTORY
        # --------------------------------------

        self.spike_events = []

        self.voltage_history = {}

        self.spike_history = {}

        self.time_history = []

        self.pending_events = {}

        # --------------------------------------
        # INITIALIZE HISTORY
        # --------------------------------------

        for neuron in self.neurons.values():

            self.voltage_history[
                neuron.name
            ] = []

            self.spike_history[
                neuron.name
            ] = []


    # ==========================================
    # SCHEDULE SYNAPTIC EVENT
    # ==========================================

    def _schedule_event(
        self,
        delivery_time,
        synapse
    ):

        # --------------------------------------
        # TRANSMIT THROUGH SYNAPSE
        # --------------------------------------

        transmission = synapse.transmit(
            delivery_time
        )

        signal = transmission[
            "signal"
        ]

        # --------------------------------------
        # CREATE EVENT QUEUE
        # --------------------------------------

        if delivery_time not in self.pending_events:

            self.pending_events[
                delivery_time
            ] = []

        self.pending_events[
            delivery_time
        ].append(
            (
                synapse.post_neuron,
                signal
            )
        )

        # --------------------------------------
        # RECORD EVENT
        # --------------------------------------

        self.spike_events.append(
            {
                "time": delivery_time,
                "pre": synapse.pre_neuron.name,
                "post": synapse.post_neuron.name,
                "signal": signal
            }
        )


    # ==========================================
    # DELIVER SYNAPTIC EVENTS
    # ==========================================

    def _deliver_events(
        self,
        current_time
    ):

        events = self.pending_events.pop(
            current_time,
            []
        )

        # --------------------------------------
        # DELIVER ALL EVENTS AT THIS TIME
        # --------------------------------------

        for neuron, signal in events:

            neuron.synaptic_events.append(
                (
                    current_time,
                    signal
                )
            )

            neuron.externally_driven = True


    # ==========================================
    # RUN NETWORK
    # ==========================================

    def run(self):

        print()
        print("======================================")
        print("RUNNING CONVERGENT NETWORK")
        print("======================================")

        # --------------------------------------
        # RESET
        # --------------------------------------

        self.reset()

        # --------------------------------------
        # BUILD OUTGOING MAP
        # --------------------------------------

        outgoing = (
            self._build_outgoing_map()
        )

        # --------------------------------------
        # FIND SOURCE NEURONS
        # --------------------------------------

        source_neurons = (
            self._find_source_neurons()
        )

        # --------------------------------------
        # VALIDATE NETWORK
        # --------------------------------------

        if len(source_neurons) == 0:

            raise ValueError(
                "Network has no source neurons."
            )

        print()
        print("Source neurons:")

        for neuron in source_neurons:

            print(
                f"  {neuron.name}"
            )

        # --------------------------------------
        # GLOBAL CLOCK
        # --------------------------------------

        current_time = 0.0

        # ======================================
        # MAIN SIMULATION LOOP
        # ======================================

        while current_time <= self.simulation_time:

            # ==================================
            # RECORD TIME
            # ==================================

            self.time_history.append(
                current_time
            )

            # ==================================
            # DELIVER SYNAPTIC EVENTS
            # ==================================

            self._deliver_events(
                current_time
            )

            # ==================================
            # STEP EVERY NEURON
            # ==================================

            newly_spiking = []

            for neuron in self.neurons.values():

                # ----------------------------------
                # SPIKES BEFORE STEP
                # ----------------------------------

                previous_spike_count = len(
                    neuron.spike_times
                )

                # ----------------------------------
                # SOURCE NEURONS CAN SPIKE
                # INTRINSICALLY.
                #
                # DOWNSTREAM NEURONS CANNOT.
                # ----------------------------------

                allow_intrinsic_trigger = (
                    neuron in source_neurons
                )

                # ----------------------------------
                # ADVANCE ONE STEP
                # ----------------------------------

                neuron.step(
                    allow_intrinsic_trigger=(
                        allow_intrinsic_trigger
                    )
                )

                # ----------------------------------
                # RECORD VOLTAGE
                # ----------------------------------

                self.voltage_history[
                    neuron.name
                ].append(
                    neuron.V
                )

                # ----------------------------------
                # CHECK FOR NEW SPIKES
                # ----------------------------------

                if (
                    len(neuron.spike_times)
                    >
                    previous_spike_count
                ):

                    new_spikes = (
                        neuron.spike_times[
                            previous_spike_count:
                        ]
                    )

                    for spike_time in new_spikes:

                        newly_spiking.append(
                            (
                                neuron,
                                spike_time
                            )
                        )

            # ==================================
            # PROPAGATE NEW SPIKES
            # ==================================

            for neuron, spike_time in newly_spiking:

                # ----------------------------------
                # RECORD SPIKE
                # ----------------------------------

                self.spike_history[
                    neuron.name
                ].append(
                    spike_time
                )

                # ----------------------------------
                # SEND TO EVERY DOWNSTREAM
                # SYNAPSE
                # ----------------------------------

                for synapse in outgoing[
                    neuron
                ]:

                    delivery_time = (
                        current_time
                        + self.dt
                    )

                    self._schedule_event(
                        delivery_time,
                        synapse
                    )

            # ==================================
            # ADVANCE GLOBAL CLOCK
            # ==================================

            current_time += self.dt

        # ======================================
        # COMPLETE
        # ======================================

        print()
        print("======================================")
        print("CONVERGENT NETWORK COMPLETE")
        print("======================================")


    # ==========================================
    # PRINT RESULTS
    # ==========================================

    def print_results(self):

        print()
        print("======================================")
        print("NETWORK RESULTS")
        print("======================================")

        # --------------------------------------
        # NEURON RESULTS
        # --------------------------------------

        for neuron in self.neurons.values():

            print()
            print(
                neuron.name
            )

            print()
            print("Spike times:")

            print(
                self.spike_history[
                    neuron.name
                ]
            )

            print()
            print("Final voltage:")

            print(
                neuron.V
            )

            print()
            print("Final phase:")

            print(
                neuron.phase
            )

        # ======================================
        # SYNAPTIC EVENTS
        # ======================================

        print()
        print("======================================")
        print("SYNAPTIC EVENTS")
        print("======================================")

        for event in self.spike_events:

            print(
                f"{event['pre']} → "
                f"{event['post']} | "
                f"{event['time']:.4f} ms | "
                f"signal = "
                f"{event['signal']:.4f}"
            )


# ==========================================
# DIRECT TEST
#
# CONVERGENT NETWORK:
#
#        ┌──────→ N3 ──────→ N4
#        │
# N1 ────┤
#        │
# N2 ────┘
# ==========================================

if __name__ == "__main__":

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

    N4 = Neuron(
        name="N4"
    )

    # ======================================
    # CREATE NETWORK
    # ======================================

    network = SharedClockNetwork(
        simulation_time=50.0,
        dt=0.01
    )

    # ======================================
    # ADD NEURONS
    # ======================================

    network.add_neuron(N1)

    network.add_neuron(N2)

    network.add_neuron(N3)

    network.add_neuron(N4)

    # ======================================
    # CONVERGENT CONNECTION 1
    #
    # N1 → N3
    # ======================================

    network.connect(
        N1,
        N3,
        weight=5.0
    )

    # ======================================
    # CONVERGENT CONNECTION 2
    #
    # N2 → N3
    # ======================================

    network.connect(
        N2,
        N3,
        weight=5.0
    )

    # ======================================
    # DOWNSTREAM CONNECTION
    #
    # N3 → N4
    # ======================================

    network.connect(
        N3,
        N4,
        weight=5.0
    )

    # ======================================
    # RUN
    # ======================================

    network.run()

    # ======================================
    # RESULTS
    # ======================================

    network.print_results()