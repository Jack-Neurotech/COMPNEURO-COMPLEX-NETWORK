# ==========================================
# NETWORK NEURON
# SIX-PHASE ACTION POTENTIAL
# WITH PERSISTENT SYNAPTIC INPUT
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
# MEMBRANE / SIMULATION PARAMETERS
# ==========================================

R = 10.0
tau_m = 10.0
C_m = 1.0

dt = 0.01
simulation_time = 50.0


# ==========================================
# SYNAPTIC PARAMETERS
# ==========================================

# How long a synaptic current remains active
# after a presynaptic spike arrives.

SYNAPTIC_DURATION = 5.0


# ==========================================
# NEURON SIMULATION
# ==========================================

def simulate_neuron(synaptic_events=None):

    if synaptic_events is None:
        synaptic_events = []


    # --------------------------------------
    # INITIAL CONDITIONS
    # --------------------------------------

    V = V_rest

    t = 0.0


    # --------------------------------------
    # STORE RESULTS
    # --------------------------------------

    times = []
    voltages = []


    # --------------------------------------
    # SPIKE STATE
    # --------------------------------------

    spike_triggered = False


    # ======================================
    # SIMULATION LOOP
    # ======================================

    while t <= simulation_time:


        # ==================================
        # SYNAPTIC CURRENT
        # ==================================

        I_syn = 0.0


        # ----------------------------------
        # PROCESS SYNAPTIC EVENTS
        # ----------------------------------

        # Each event has:
        #
        #     event_time
        #     event_current
        #
        # The current remains active for
        # SYNAPTIC_DURATION milliseconds.

        for event_time, event_current in synaptic_events:

            if (
                t >= event_time
                and
                t < event_time + SYNAPTIC_DURATION
            ):

                I_syn += event_current


        # ==================================
        # PHASE 1
        # SUBTHRESHOLD
        # ==================================

        if not spike_triggered:

            I_input = 1.5

            I_total = (
                I_input
                + I_syn
            )


            dVdt = (
                -(V - E_L)
                + R * I_total
            ) / tau_m


            V = V + dVdt * dt


            # ----------------------------------
            # THRESHOLD DETECTION
            # ----------------------------------

            if V >= V_threshold:

                spike_triggered = True

                V = V_threshold


        # ==================================
        # PHASE 2
        # DEPOLARIZATION
        # ==================================

        elif t < 21.0:

            I_Na = 100.0

            V = V + (
                I_Na / C_m
            ) * dt


            if V > V_peak:

                V = V_peak


        # ==================================
        # PHASE 3
        # PEAK
        # ==================================

        elif t < 21.5:

            I_Na = 0.0
            I_K = 0.0


            V = V + (
                (I_Na + I_K) / C_m
            ) * dt


            V = V_peak


        # ==================================
        # PHASE 4
        # REPOLARIZATION
        # ==================================

        elif t < 23.0:

            I_Na = 0.0
            I_K = -70.0


            V = V + (
                I_K / C_m
            ) * dt


            if V < -70.0:

                V = -70.0


        # ==================================
        # PHASE 5
        # HYPERPOLARIZATION
        # ==================================

        elif t < 28.0:

            I_Na = 0.0
            I_K = -1.0


            V = V + (
                I_K / C_m
            ) * dt


            if V < V_hyper:

                V = V_hyper


        # ==================================
        # PHASE 6
        # RETURN TO BASELINE
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


            V = V + dVdt * dt


        # ==================================
        # STORE CURRENT STATE
        # ==================================

        times.append(t)

        voltages.append(V)


        # Advance simulation clock.

        t += dt


    # ======================================
    # RETURN SIMULATION
    # ======================================

    return (
        np.array(times),
        np.array(voltages)
    )