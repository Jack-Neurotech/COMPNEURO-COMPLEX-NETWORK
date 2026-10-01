
# ==========================================
# SYNAPSE
# API-SAFE SYNAPTIC TRANSMISSION + STDP
# ==========================================

from dataclasses import dataclass
import math

from stdp_models import (
    STDPParameters,
    weight_change
)


# ==========================================
# STDP RESULT
# ==========================================

@dataclass(frozen=True)
class STDPResult:
    """
    Result returned by a synaptic STDP update.
    """

    pre_spike_time: float
    post_spike_time: float
    delta_t: float
    delta_w: float
    old_weight: float
    new_weight: float


# ==========================================
# SYNAPSE
# ==========================================

class Synapse:

    def __init__(
        self,
        pre_neuron,
        post_neuron,
        weight=1.0
    ):

        self._validate_weight(weight)

        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = float(weight)

    # ==========================================
    # VALIDATION
    # ==========================================

    @staticmethod
    def _validate_weight(weight):

        if not isinstance(
            weight,
            (int, float)
        ):

            raise TypeError(
                "weight must be a number."
            )

        if not math.isfinite(weight):

            raise ValueError(
                "weight must be finite."
            )

    @staticmethod
    def _validate_spike_time(
        spike_time,
        name
    ):

        if not isinstance(
            spike_time,
            (int, float)
        ):

            raise TypeError(
                f"{name} must be a number."
            )

        if not math.isfinite(spike_time):

            raise ValueError(
                f"{name} must be finite."
            )

    # ==========================================
    # SYNAPTIC TRANSMISSION
    # ==========================================

    def transmit(self, spike_time):

        self._validate_spike_time(
            spike_time,
            "spike_time"
        )

        return {
            "spike_time": float(spike_time),
            "signal": self.weight
        }

    # ==========================================
    # STDP UPDATE
    # ==========================================

    def apply_stdp(
        self,
        pre_spike_time,
        post_spike_time,
        parameters=None
    ):

        self._validate_spike_time(
            pre_spike_time,
            "pre_spike_time"
        )

        self._validate_spike_time(
            post_spike_time,
            "post_spike_time"
        )

        if parameters is None:

            parameters = STDPParameters()

        if not isinstance(
            parameters,
            STDPParameters
        ):

            raise TypeError(
                "parameters must be an "
                "STDPParameters instance."
            )

        delta_t = (
            float(post_spike_time)
            - float(pre_spike_time)
        )

        delta_w = weight_change(
            delta_t,
            parameters
        )

        old_weight = self.weight

        new_weight = (
            old_weight
            + delta_w
        )

        self._validate_weight(
            new_weight
        )

        self.weight = new_weight

        return STDPResult(
            pre_spike_time=float(
                pre_spike_time
            ),
            post_spike_time=float(
                post_spike_time
            ),
            delta_t=float(
                delta_t
            ),
            delta_w=float(
                delta_w
            ),
            old_weight=float(
                old_weight
            ),
            new_weight=float(
                new_weight
            )
        )


# ==========================================
# DIRECT API TEST
# ==========================================

if __name__ == "__main__":

    synapse = Synapse(
        pre_neuron="N1",
        post_neuron="N2",
        weight=5.0
    )

    print("======================================")
    print("SYNAPSE API TEST")
    print("======================================")

    transmission = synapse.transmit(
        spike_time=20.0
    )

    print()
    print("Transmission:")
    print(transmission)

    print()
    print("Initial weight:")
    print(synapse.weight)

    result = synapse.apply_stdp(
        pre_spike_time=20.0,
        post_spike_time=21.0
    )

    print()
    print("STDP result:")
    print(result)

    print()
    print("Delta t:")
    print(result.delta_t)

    print()
    print("Delta w:")
    print(result.delta_w)

    print()
    print("Old weight:")
    print(result.old_weight)

    print()
    print("New weight:")
    print(result.new_weight)
