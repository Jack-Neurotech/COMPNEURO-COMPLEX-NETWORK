# ==========================================
# NEURAL RECORDING ADAPTER
#
# Converts Project 1 neuron outputs into
# analysis-ready population signals.
#
# SOURCE:
#   neuron.py
#
# PURPOSE:
#   Preserve neuron.py exactly while creating
#   a clean bridge into the recycled EEG-style
#   signal-analysis components.
#
# IMPORTANT:
#   These are simulated neural signals, NOT
#   human EEG measurements.
# ==========================================

import numpy as np


class NeuralRecording:

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(self, neurons):

        if not neurons:
            raise ValueError(
                "NeuralRecording requires at least one neuron."
            )

        self.neurons = list(neurons)

        self.channel_names = [
            neuron.name
            for neuron in self.neurons
        ]

        self.times = None
        self.signals = None
        self.sampling_interval = None
        self.sampling_rate = None


    # ==========================================
    # READ NEURON OUTPUTS
    #
    # Reads:
    #   neuron.times
    #   neuron.voltages
    #
    # This does not modify the neurons.
    # ==========================================

    def read(self):

        recordings = []

        reference_times = None

        for neuron in self.neurons:

            times = np.asarray(
                neuron.times,
                dtype=float
            )

            voltages = np.asarray(
                neuron.voltages,
                dtype=float
            )

            if len(times) == 0:
                raise ValueError(
                    f"{neuron.name} has no recorded time points."
                )

            if len(times) != len(voltages):
                raise ValueError(
                    f"{neuron.name} has mismatched "
                    "time and voltage arrays."
                )

            if reference_times is None:

                reference_times = times

            elif not np.allclose(
                reference_times,
                times
            ):

                raise ValueError(
                    "All neurons must share the same "
                    "simulation time base."
                )

            recordings.append(
                voltages
            )

        self.times = reference_times
        self.signals = np.vstack(recordings)

        # --------------------------------------
        # TIME / SAMPLING METADATA
        # --------------------------------------

        if len(self.times) > 1:

            self.sampling_interval = (
                self.times[1]
                - self.times[0]
            )

            if self.sampling_interval <= 0:

                raise ValueError(
                    "Simulation time must increase."
                )

            self.sampling_rate = (
                1.0
                /
                self.sampling_interval
            )

        return self.signals


    # ==========================================
    # GET SINGLE NEURON SIGNAL
    # ==========================================

    def get_signal(self, neuron_name):

        if self.signals is None:

            self.read()

        if neuron_name not in self.channel_names:

            raise KeyError(
                f"Unknown neuron: {neuron_name}"
            )

        index = self.channel_names.index(
            neuron_name
        )

        return self.signals[index]


    # ==========================================
    # POPULATION MEAN SIGNAL
    #
    # Produces one signal representing the
    # average membrane activity across the
    # recorded population.
    # ==========================================

    def population_mean(self):

        if self.signals is None:

            self.read()

        return np.mean(
            self.signals,
            axis=0
        )


    # ==========================================
    # POPULATION SUM SIGNAL
    #
    # Produces the summed population activity.
    # ==========================================

    def population_sum(self):

        if self.signals is None:

            self.read()

        return np.sum(
            self.signals,
            axis=0
        )


    # ==========================================
    # POPULATION SUMMARY
    # ==========================================

    def summary(self):

        if self.signals is None:

            self.read()

        return {

            "number_of_neurons":
                len(self.neurons),

            "number_of_samples":
                self.signals.shape[1],

            "simulation_duration":
                self.times[-1]
                - self.times[0],

            "sampling_interval":
                self.sampling_interval,

            "sampling_rate":
                self.sampling_rate,

            "channel_names":
                self.channel_names.copy(),

        }


# ==========================================
# DIRECT TEST
# ==========================================

if __name__ == "__main__":

    from neuron import Neuron


    # --------------------------------------
    # CREATE TEST POPULATION
    # --------------------------------------

    neurons = [

        Neuron("N1"),

        Neuron("N2"),

        Neuron("N3"),

    ]


    # --------------------------------------
    # RUN EACH NEURON
    # --------------------------------------

    for neuron in neurons:

        neuron.simulate()


    # --------------------------------------
    # CREATE RECORDING
    # --------------------------------------

    recording = NeuralRecording(
        neurons
    )


    signals = recording.read()


    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    print("======================================")
    print("NEURAL RECORDING TEST")
    print("======================================")

    print()
    print("Channels:")
    print(recording.channel_names)

    print()
    print("Signal shape:")
    print(signals.shape)

    print()
    print("Sampling interval:")
    print(recording.sampling_interval)

    print()
    print("Sampling rate:")
    print(recording.sampling_rate)

    print()
    print("Population mean shape:")
    print(
        recording.population_mean().shape
    )

    print()
    print("Summary:")
    print(
        recording.summary()
    )
