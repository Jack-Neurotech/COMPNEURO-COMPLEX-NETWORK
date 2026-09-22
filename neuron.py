# ==========================================
# REUSABLE NETWORK NEURON
#
# SIX-PHASE ACTION POTENTIAL
# STATEFUL STEP-BASED VERSION
#
# SOURCE OF TRUTH:
# ORIGINAL SIX-PHASE NEURON MODEL
# ==========================================

import numpy as np


class Neuron:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        name,
        synaptic_duration=5.0
    ):

        # --------------------------------------
        # IDENTITY
        # --------------------------------------

        self.name = name

        # --------------------------------------
        # MEMBRANE PARAMETERS
        # --------------------------------------

        self.E_L = -70.0
        self.V_rest = -70.0
        self.V_threshold = -55.0
        self.V_peak = 30.0
        self.V_hyper = -75.0

        # --------------------------------------
        # MEMBRANE CONSTANTS
        # --------------------------------------

        self.R = 10.0
        self.tau_m = 10.0
        self.C_m = 1.0

        # --------------------------------------
        # SIMULATION PARAMETERS
        # --------------------------------------

        self.dt = 0.01
        self.simulation_time = 50.0

        # --------------------------------------
        # SIX-PHASE TIMINGS
        # --------------------------------------

        self.phase_1_duration = 20.0
        self.phase_2_duration = 1.0
        self.phase_3_duration = 0.5
        self.phase_4_duration = 1.5
        self.phase_5_duration = 5.0

        # --------------------------------------
        # SYNAPTIC PARAMETERS
        # --------------------------------------

        self.synaptic_duration = synaptic_duration

        # --------------------------------------
        # RESET EVERYTHING
        # --------------------------------------

        self.reset_state()


    # ==========================================
    # RESET STATE
    # ==========================================

    def reset_state(self):

        # --------------------------------------
        # MEMBRANE STATE
        # --------------------------------------

        self.V = self.V_rest
        self.t = 0.0

        # --------------------------------------
        # PHASE STATE
        # --------------------------------------

        self.phase = "subthreshold"

        # --------------------------------------
        # ACTION POTENTIAL STATE
        # --------------------------------------

        self.action_potential_start = None

        # --------------------------------------
        # SPIKE STATE
        # --------------------------------------

        self.spike_detected = False
        self.spike_times = []

        # --------------------------------------
        # SYNAPTIC INPUT
        # --------------------------------------

        self.synaptic_events = []
        self.externally_driven = False

        # --------------------------------------
        # TRAJECTORY
        # --------------------------------------

        self.times = []
        self.voltages = []


    # ==========================================
    # SET SYNAPTIC EVENTS
    # ==========================================

    def set_synaptic_events(
        self,
        synaptic_events=None
    ):

        if synaptic_events is None:
            synaptic_events = []

        self.synaptic_events = list(
            synaptic_events
        )

        self.externally_driven = (
            len(self.synaptic_events) > 0
        )


    # ==========================================
    # GET SYNAPTIC CURRENT
    # ==========================================

    def _get_synaptic_current(self):

        I_syn = 0.0

        for event_time, event_current in (
            self.synaptic_events
        ):

            if (
                self.t >= event_time
                and
                self.t <
                event_time + self.synaptic_duration
            ):

                I_syn += event_current

        return I_syn


    # ==========================================
    # ONE SIMULATION STEP
    #
    # allow_intrinsic_trigger:
    #
    # True:
    #     neuron can initiate its own spike
    #
    # False:
    #     neuron must wait for synaptic input
    # ==========================================

    def step(
        self,
        allow_intrinsic_trigger=True
    ):

        # ======================================
        # CURRENT SYNAPTIC INPUT
        # ======================================

        I_syn = self._get_synaptic_current()


        # ======================================
        # PHASE 1
        #
        # SUBTHRESHOLD / RESTING
        # ======================================

        if self.phase == "subthreshold":

            I_input = 1.5

            I_total = (
                I_input
                + I_syn
            )

            dVdt = (
                -(self.V - self.E_L)
                + self.R * I_total
            ) / self.tau_m

            self.V = (
                self.V
                + dVdt * self.dt
            )

            # ----------------------------------
            # SYNAPTICALLY DRIVEN SPIKE
            # ----------------------------------

            if (
                self.externally_driven
                and
                self.V >= self.V_threshold
            ):

                self.phase = "depolarization"

                self.action_potential_start = (
                    self.t
                )

                if not self.spike_detected:

                    self.spike_detected = True

                    self.spike_times.append(
                        self.t
                    )

            # ----------------------------------
            # NORMAL INTRINSIC TRIGGER
            # ----------------------------------

            elif (
                allow_intrinsic_trigger
                and
                not self.externally_driven
                and
                self.t >= self.phase_1_duration
            ):

                self.phase = "depolarization"

                self.action_potential_start = (
                    self.t
                )

        # ======================================
        # PHASE 2
        #
        # DEPOLARIZATION
        # ======================================

        elif self.phase == "depolarization":

            elapsed = (
                self.t
                -
                self.action_potential_start
            )

            if elapsed < self.phase_2_duration:

                I_Na = 100.0

                self.V = (
                    self.V
                    +
                    (
                        I_Na
                        /
                        self.C_m
                    )
                    * self.dt
                )

                if self.V > self.V_peak:

                    self.V = self.V_peak

                # ----------------------------------
                # SPIKE DETECTION
                # ----------------------------------

                if (
                    not self.spike_detected
                    and
                    self.V >= self.V_threshold
                ):

                    self.spike_detected = True

                    self.spike_times.append(
                        self.t
                    )

            else:

                self.phase = "peak"


        # ======================================
        # PHASE 3
        #
        # PEAK
        # ======================================

        elif self.phase == "peak":

            elapsed = (
                self.t
                -
                self.action_potential_start
            )

            if (
                elapsed
                <
                (
                    self.phase_2_duration
                    +
                    self.phase_3_duration
                )
            ):

                I_Na = 0.0
                I_K = 0.0

                self.V = (
                    self.V
                    +
                    (
                        I_Na
                        +
                        I_K
                    )
                    /
                    self.C_m
                    *
                    self.dt
                )

                self.V = self.V_peak

            else:

                self.phase = "repolarization"


        # ======================================
        # PHASE 4
        #
        # REPOLARIZATION
        # ======================================

        elif self.phase == "repolarization":

            elapsed = (
                self.t
                -
                self.action_potential_start
            )

            if (
                elapsed
                <
                (
                    self.phase_2_duration
                    +
                    self.phase_3_duration
                    +
                    self.phase_4_duration
                )
            ):

                I_Na = 0.0
                I_K = -70.0

                self.V = (
                    self.V
                    +
                    (
                        I_K
                        /
                        self.C_m
                    )
                    *
                    self.dt
                )

                if self.V < self.V_rest:

                    self.V = self.V_rest

            else:

                self.phase = "hyperpolarization"


        # ======================================
        # PHASE 5
        #
        # HYPERPOLARIZATION
        # ======================================

        elif self.phase == "hyperpolarization":

            elapsed = (
                self.t
                -
                self.action_potential_start
            )

            if (
                elapsed
                <
                (
                    self.phase_2_duration
                    +
                    self.phase_3_duration
                    +
                    self.phase_4_duration
                    +
                    self.phase_5_duration
                )
            ):

                I_Na = 0.0
                I_K = -1.0

                self.V = (
                    self.V
                    +
                    (
                        I_K
                        /
                        self.C_m
                    )
                    *
                    self.dt
                )

                if self.V < self.V_hyper:

                    self.V = self.V_hyper

            else:

                self.phase = "baseline"


        # ======================================
        # PHASE 6
        #
        # RETURN TO BASELINE
        # ======================================

        elif self.phase == "baseline":

            I_input = 0.0

            I_total = (
                I_input
                + I_syn
            )

            dVdt = (
                -(self.V - self.E_L)
                + self.R * I_total
            ) / self.tau_m

            self.V = (
                self.V
                +
                dVdt * self.dt
            )


        # ======================================
        # RECORD CURRENT STATE
        # ======================================

        self.times.append(self.t)

        self.voltages.append(self.V)


        # ======================================
        # ADVANCE TIME
        # ======================================

        self.t += self.dt


        # ======================================
        # RETURN VOLTAGE
        # ======================================

        return self.V


    # ==========================================
    # FULL SIMULATION
    #
    # BUILT ENTIRELY FROM step()
    # ==========================================

    def simulate(
        self,
        synaptic_events=None
    ):

        # --------------------------------------
        # RESET
        # --------------------------------------

        self.reset_state()

        # --------------------------------------
        # LOAD SYNAPTIC EVENTS
        # --------------------------------------

        self.set_synaptic_events(
            synaptic_events
        )

        # --------------------------------------
        # RUN
        # --------------------------------------

        while self.t <= self.simulation_time:

            self.step(
                allow_intrinsic_trigger=True
            )

        # --------------------------------------
        # CONVERT TO NUMPY
        # --------------------------------------

        self.times = np.array(
            self.times
        )

        self.voltages = np.array(
            self.voltages
        )

        return (
            self.times,
            self.voltages
        )


# ==========================================
# DIRECT TEST
# ==========================================

if __name__ == "__main__":

    neuron = Neuron(
        name="N1"
    )

    neuron.simulate()

    print("======================================")
    print("STATEFUL NEURON TEST")
    print("======================================")

    print()
    print("Neuron:")
    print(neuron.name)

    print()
    print("Minimum voltage:")
    print(neuron.voltages.min())

    print()
    print("Maximum voltage:")
    print(neuron.voltages.max())

    print()
    print("Spike times:")
    print(neuron.spike_times)

    print()
    print("Final phase:")
    print(neuron.phase)

    print()
    print("Final time:")
    print(neuron.t)