
# ==========================================
# NETWORK NEURON
# SIX-PHASE ACTION POTENTIAL
#
# SUPPORTS:
# 1. INTRINSICALLY DRIVEN NEURONS
# 2. SYNAPTICALLY DRIVEN NEURONS
#
# N1 can fire from intrinsic input.
# N2 can fire from incoming synaptic input.
# ==========================================

import numpy as np


# ==========================================
# MEMBRANE PARAMETERS
# ==========================================

E_L = -70.0
V_rest = -70.0
V_threshold = -55.0
V_peak = 30.0
V_hyper = -75.0


# ==========================================
# MEMBRANE PARAMETERS
# ==========================================

R = 10.0
tau_m = 10.0
C_m = 1.0


# ==========================================
# SIMULATION PARAMETERS
# ==========================================

dt = 0.01
simulation_time = 50.0


# ==========================================
# SYNAPTIC PARAMETERS
# ==========================================

SYNAPTIC_DURATION = 5.0


# ==========================================
# NEURON SIMULATION
# ==========================================

def simulate_neuron(
    synaptic_events=None,
    intrinsic_input=True
):

    # ======================================
    # DEFAULT SYNAPTIC EVENTS
    # ======================================

    if synaptic_events is None:

        synaptic_events = []


    # ======================================
    # INITIAL CONDITIONS
    # ======================================

    V = V_rest

    t = 0.0


    # ======================================
    # STORE RESULTS
    # ======================================

    times = []

    voltages = []


    # ======================================
    # SPIKE STATE
    # ======================================

    spike_triggered = False


    # ======================================
    # SIMULATION LOOP
    # ======================================

    while t <= simulation_time:


        # ==================================
        # SYNAPTIC CURRENT
        # ==================================

        I_syn = 0.0


        for event_time, event_current in (
            synaptic_events
        ):

            if (
                t >= event_time
                and
                t <
                event_time
                + SYNAPTIC_DURATION
            ):

                I_syn += event_current


        # ==================================
        # PHASE 1
        # SUBTHRESHOLD
        #
        # Before the action potential,
        # membrane voltage changes according
        # to intrinsic and/or synaptic input.
        # ==================================

        if not spike_triggered:

            # ----------------------------------
            # INTRINSIC INPUT
            # ----------------------------------

            if intrinsic_input:

                I_input = 1.5

            else:

                I_input = 0.0


            # ----------------------------------
            # TOTAL CURRENT
            # ----------------------------------

            I_total = (
                I_input
                + I_syn
            )


            # ----------------------------------
            # MEMBRANE EQUATION
            # ----------------------------------

            dVdt = (
                -(V - E_L)
                + R * I_total
            ) / tau_m


            V = (
                V
                + dVdt * dt
            )


            # ----------------------------------
            # THRESHOLD DETECTION
            # ----------------------------------

            if V >= V_threshold:

                spike_triggered = True

                V = V_threshold


        # ==================================
        # PHASE 2
        # DEPOLARIZATION
        #
        # 1 ms action-potential rise
        # ==================================

        elif t < 21.0:

            I_Na = 100.0


            V = (
                V
                + (
                    I_Na
                    /
                    C_m
                )
                * dt
            )


            if V > V_peak:

                V = V_peak


        # ==================================
        # PHASE 3
        # PEAK
        #
        # 21 → 21.5 ms
        # ==================================

        elif t < 21.5:

            I_Na = 0.0

            I_K = 0.0


            V = (
                V
                + (
                    (
                        I_Na
                        +
                        I_K
                    )
                    /
                    C_m
                )
                * dt
            )


            V = V_peak


        # ==================================
        # PHASE 4
        # REPOLARIZATION
        #
        # 21.5 → 23 ms
        # ==================================

        elif t < 23.0:

            I_Na = 0.0

            I_K = -70.0


            V = (
                V
                + (
                    I_K
                    /
                    C_m
                )
                * dt
            )


            if V < V_rest:

                V = V_rest


        # ==================================
        # PHASE 5
        # HYPERPOLARIZATION
        #
        # 23 → 28 ms
        # ==================================

        elif t < 28.0:

            I_Na = 0.0

            I_K = -1.0


            V = (
                V
                + (
                    I_K
                    /
                    C_m
                )
                * dt
            )


            if V < V_hyper:

                V = V_hyper


        # ==================================
        # PHASE 6
        # RETURN TO BASELINE
        #
        # 28 → 50 ms
        # ==================================

        else:

            I_input = 0.0


            I_total = (
                I_input
                + I_syn
            )


            dVdt = (
                -(V - E_L)
                + R * I_total
            ) / tau_m


            V = (
                V
                + dVdt * dt
            )


        # ==================================
        # STORE STATE
        # ==================================

        times.append(t)

        voltages.append(V)


        # ==================================
        # ADVANCE TIME
        # ==================================

        t += dt


    # ======================================
    # CONVERT TO NUMPY ARRAYS
    # ======================================

    times = np.array(times)

    voltages = np.array(voltages)


    # ======================================
    # DETECT SPIKES
    # ======================================

    spike_times = []


    for i in range(1, len(voltages)):

        previous_voltage = (
            voltages[i - 1]
        )

        current_voltage = (
            voltages[i]
        )


        if (
            previous_voltage
            < V_threshold
            and
            current_voltage
            >= V_threshold
        ):

            spike_times.append(
                times[i]
            )


    # ======================================
    # RETURN RESULTS
    # ======================================

    return (
        times,
        voltages,
        spike_times
    )
