# COMPNEURO-COMPLEX-NETWORK

# 1. What Is the Project and What Does It Do?

* **Project:** Computational Neuroscience Complex Neural Network Simulation

* **Purpose:** A Python-based computational neuroscience project that builds a reusable model of neurons, synapses, neural networks, synaptic transmission, spike timing, and spike-timing-dependent plasticity (STDP).

* **Primary goal:** Simulate how individual neurons generate action potentials, how neurons communicate through synapses, how activity propagates through networks, and how synaptic connections change through repeated spike-timing-dependent learning.

* **The project is organized into several experimental areas:**

  * Communication experiments
  * Network experiments
  * Learning experiments
  * Visualization files
  * Core reusable neural components
  * Legacy implementations

* **The core computational components are:**

  * `neuron.py`
  * `network.py`
  * `synapse.py`
  * `stdp_models.py`

* **`neuron.py` provides the neuron model:**

  * Represents an individual neuron as a Python object.
  * Maintains membrane voltage.
  * Maintains simulation time.
  * Maintains the neuron's current physiological phase.
  * Generates a six-phase action potential.
  * Detects spikes.
  * Records spike times.
  * Accepts synaptic input.
  * Can operate as an intrinsically firing neuron.
  * Can operate as a neuron driven by synaptic input.
  * Provides a reusable `step()` method for advancing the neuron one simulation step.
  * Provides a `simulate()` method for running the complete neuron simulation.

* **The six-phase neuron model represents:**

  * Subthreshold activity
  * Depolarization
  * Peak
  * Repolarization
  * Hyperpolarization
  * Return to baseline

* **`network.py` provides the network model:**

  * Stores neurons.
  * Stores synapses.
  * Connects neurons together.
  * Identifies source neurons.
  * Builds outgoing connection maps.
  * Detects spike events.
  * Propagates spike events through connected neurons.
  * Supports multi-hop propagation through network layers.

* **`synapse.py` provides synaptic communication and plasticity:**

  * Connects a presynaptic neuron to a postsynaptic neuron.
  * Stores synaptic weight.
  * Transmits spike events.
  * Calculates the difference between pre- and postsynaptic spike times.
  * Applies STDP.
  * Updates synaptic weight.
  * Validates synaptic weights and spike times.

* **`stdp_models.py` provides the mathematical learning model:**

  * Defines STDP parameters.
  * Defines potentiation parameters.
  * Defines depression parameters.
  * Calculates synaptic weight changes from spike-time differences.
  * Uses an exponential learning window.
  * Produces potentiation when the presynaptic neuron fires before the postsynaptic neuron.
  * Produces depression when the postsynaptic neuron fires before the presynaptic neuron.

* **The communication experiments demonstrate:**

  * A neuron generating a spike.
  * A spike being transmitted through a synapse.
  * A downstream neuron receiving synaptic input.
  * The relationship between spike timing and synaptic transmission.
  * Basic two-neuron communication.
  * The transition from an individual neuron model into an interacting neural system.

* **The communication branch contains experiments including:**

  * `two_neuron_communication.py`
  * `two_neuron_network.py`
  * Reusable neuron/network foundation files
  * Synapse implementation
  * Experiment runner files

* **The network experiments demonstrate:**

  * Multiple neurons operating as a connected system.
  * Chain networks.
  * Branching networks.
  * Spike propagation from one neuron to another.
  * Multi-hop communication across network layers.
  * How network topology determines the path through which neural activity can propagate.

* **The network branch contains:**

  * `branching_network.py`
  * `chain_network.py`
  * Experiment runner files.

* **The learning experiments demonstrate:**

  * Repeated neural network trials.
  * Spike-time measurement.
  * Calculation of pre/post spike-time differences.
  * STDP-based synaptic weight updates.
  * Persistent synaptic learning across trials.
  * Separate learning of different synapses.
  * Comparison of learning across branching and chain network structures.

* **The branching learning experiment specifically:**

  * Creates N1, N2, and N3.
  * Connects N1 to N2.
  * Connects N1 to N3.
  * Runs repeated trials.
  * Resets neuron state between trials.
  * Preserves synaptic weights between trials.
  * Measures spike timing.
  * Calculates STDP.
  * Updates both synaptic weights.
  * Stores weight and spike-timing histories.

* **The learning experiments include:**

  * `branching_learning.py`
  * `chain_learning.py`
  * `stdp_learning_experiment.py`
  * `stdp_network_experiment.py`

* **The visualization files communicate the learning process visually:**

  * Synaptic weight changes across trials.
  * Spike-time differences across trials.
  * Spike timing across trials.
  * Learning curves generated from repeated STDP experiments.

* **The visualization implementation produces three primary analyses:**

  * Synaptic weight vs. trial.
  * Spike-time difference vs. trial.
  * Spike timing vs. trial.

* **The experiments therefore communicate several levels of computational neuroscience:**

  * **Level 1 — Neuron:** How an individual neuron changes voltage and generates an action potential.
  * **Level 2 — Synapse:** How one neuron's spike becomes an input to another neuron.
  * **Level 3 — Communication:** How neural signals move between neurons.
  * **Level 4 — Network:** How multiple connected neurons form chains and branching structures.
  * **Level 5 — Timing:** How the relative timing of spikes affects synaptic interactions.
  * **Level 6 — Plasticity:** How synaptic weights change according to spike timing.
  * **Level 7 — Learning:** How those changes accumulate across repeated trials.
  * **Level 8 — Visualization:** How the resulting changes can be quantitatively visualized across trials.

* **The overall scientific concept is:**

  * Individual neurons generate electrical activity.
  * Electrical activity produces spikes.
  * Spikes are transmitted through synapses.
  * Synapses connect neurons into networks.
  * Network activity produces downstream spikes.
  * The relative timing of those spikes determines STDP.
  * STDP changes synaptic weights.
  * Repeated trials allow those weight changes to accumulate.
  * Visualization allows the resulting learning behavior to be analyzed.

* **In simple terms:**

  * **Neuron → Spike → Synapse → Communication → Network → Spike Timing → STDP → Weight Change → Repeated Learning**

* **What the project demonstrates:**

  * Python programming
  * Object-oriented programming
  * Computational neuroscience
  * Neuron modeling
  * Action-potential simulation
  * Synaptic communication
  * Neural network construction
  * Event propagation
  * Spike detection
  * Spike-time analysis
  * Synaptic plasticity
  * STDP
  * Repeated-trial learning
  * Numerical modeling
  * Scientific visualization

# 2. Python Principles, General Structure, Packages, and Associated Principles

## A. Python Principles and the General Structure

The computational neuroscience project is built from fundamental Python programming concepts. These principles allow the neuron, synapse, network, learning, and visualization components to operate as separate but connected parts of the same computational system.

---

### Variables

* Variables store information so the program can use it later.
* The project uses variables to store membrane parameters, simulation settings, neuron states, synaptic weights, spike times, and learning results.

Example:

```python
self.V_rest = -70.0

self.V_threshold = -55.0

self.V_peak = 30.0

self.dt = 0.01
```

These variables store numerical parameters that define how the neuron behaves.

---

### Data Types

Python allows the project to work with different types of information.

Common types used include:

* **Numbers** — voltage, time, current, synaptic weight, and STDP measurements.
* **Strings** — neuron names and physiological phase names.
* **Lists** — spike times, voltage histories, and learning histories.
* **Dictionaries** — collections of neurons and network connections.
* **Objects** — neurons, synapses, and networks.
* **Dataclasses** — structured STDP results.

Example:

```python
self.name = name

self.phase = "subthreshold"

self.spike_times = []
```

The neuron stores its identity, current phase, and spike history using different Python data types.

---

### Lists

Lists store collections of values.

The neuron uses lists to record its trajectory:

```python
self.times = []

self.voltages = []

self.spike_times = []
```

The program can add new values as the simulation progresses.

For example:

```python
self.spike_times.append(
    self.t
)
```

adds a detected spike time to the neuron's spike history.

---

### Dictionaries

Dictionaries store information using keys.

The network uses a dictionary to store neurons:

```python
self.neurons = {}
```

Neurons are then stored using their names:

```python
self.neurons[
    neuron.name
] = neuron
```

This allows the network to retrieve neurons using their identifiers.

---

### Indexing

Indexing accesses a specific element inside a collection.

The learning experiments use indexing to obtain the first detected spike:

```python
spike_1 = N1.spike_times[0]

spike_2 = N2.spike_times[0]

spike_3 = N3.spike_times[0]
```

This allows the experiment to compare the timing of the neurons' first spikes.

---

### Functions

Functions organize reusable operations.

Examples include:

```python
def weight_change(
    delta_t_ms,
    parameters
):
```

and:

```python
def step(
    self,
    allow_intrinsic_trigger=True
):
```

Functions allow the project to separate individual computational operations.

The general structure is:

**Input → Processing → Output**

---

### Methods

Methods are functions that belong to objects.

The neuron contains methods such as:

```python
neuron.step()
```

```python
neuron.simulate()
```

```python
neuron.reset_state()
```

The synapse contains methods such as:

```python
synapse.transmit()
```

```python
synapse.apply_stdp()
```

The network contains methods such as:

```python
network.add_neuron()
```

```python
network.connect()
```

```python
network.run()
```

Each method performs an operation associated with its object.

---

### Classes

Classes provide blueprints for the computational components.

The project uses classes including:

```python
class Neuron:
```

```python
class Synapse:
```

```python
class Network:
```

and:

```python
@dataclass(frozen=True)
class STDPResult:
```

Each class represents a different component of the neural system.

---

### Objects

Objects are individual instances created from classes.

For example:

```python
N1 = Neuron(
    name="N1"
)

N2 = Neuron(
    name="N2"
)

N3 = Neuron(
    name="N3"
)
```

These are three separate neuron objects.

The network can then contain them:

```python
network.add_neuron(N1)

network.add_neuron(N2)

network.add_neuron(N3)
```

---

### Object State

Objects can maintain their own internal state.

The neuron maintains:

```python
self.V
```

for membrane voltage.

It maintains:

```python
self.t
```

for simulation time.

It maintains:

```python
self.phase
```

for its current action-potential phase.

It maintains:

```python
self.spike_times
```

for detected spikes.

This allows the neuron to remember what has happened during the simulation.

---

### State Transitions

The neuron changes between defined states.

For example:

```python
if self.phase == "subthreshold":
```

can transition to:

```python
self.phase = "depolarization"
```

Later, the neuron transitions through:

```text
subthreshold
→ depolarization
→ peak
→ repolarization
→ hyperpolarization
→ baseline
```

This allows the action potential to be represented as a sequence of computational states.

---

### Conditional Statements

Conditional statements allow the model to make decisions.

Example:

```python
if self.phase == "depolarization":
```

The neuron behaves differently depending on its current phase.

Conditions are also used to determine whether:

* A spike has occurred.
* Synaptic input is active.
* A neuron should fire.
* A network event has already been processed.
* A synaptic weight is valid.
* An STDP update should produce potentiation or depression.

---

### Boolean Logic

Boolean logic allows multiple conditions to be evaluated.

Example:

```python
if (
    not self.spike_detected
    and
    self.V >= self.V_threshold
):
```

The spike is detected only when both conditions are satisfied.

Boolean logic therefore connects the mathematical state of the neuron to discrete computational events.

---

### Loops

Loops allow the simulations to repeat operations.

The neuron simulation repeatedly advances time:

```python
while self.t <= self.simulation_time:

    self.step(
        allow_intrinsic_trigger=True
    )
```

The learning experiment repeatedly runs trials:

```python
for trial in range(
    1,
    NUMBER_OF_TRIALS + 1
):
```

This allows the project to simulate both continuous neural dynamics and repeated learning experiments.

---

### Encapsulation

Encapsulation means keeping related data and operations inside the object responsible for them.

The `Neuron` class controls:

* Membrane voltage.
* Simulation time.
* Action-potential phase.
* Spike detection.
* Synaptic input.

The `Synapse` class controls:

* Presynaptic neuron.
* Postsynaptic neuron.
* Synaptic weight.
* Transmission.
* STDP updates.

The `Network` class controls:

* Neurons.
* Synapses.
* Network connections.
* Event propagation.

This prevents the entire project from becoming one large block of code.

---

### Modular Programming

The project separates different responsibilities into different files.

The core branch contains:

```text
Core/
    neuron.py
    network.py
    synapse.py
    stdp_models.py
```

The experiments are separated into their own branches:

```text
Communication-Experiments
Learning-Experiments
Network-Experiments
VisualizationFiles
LegacyFiles
```

This allows the same core neuron, synapse, and network implementations to be reused by multiple experiments.

---

### Data Flow

Data flow describes how information moves through the computational system.

The project follows a general pattern:

**Neuron State → Spike → Synapse → Transmission → Downstream Neuron → New Spike → STDP → Weight Update**

For repeated learning:

**Trial → Spike Times → Δt → Δw → New Weight → Next Trial**

The learning experiment stores these results in histories:

```python
weight_12_history = []

weight_13_history = []

delta_t_12_history = []

delta_t_13_history = []
```

This allows the experiment to track how the network changes across trials.

---

### Exception Handling

Exceptions prevent invalid numerical information from entering the model.

For example, the synapse validates weights:

```python
if not isinstance(
    weight,
    (int, float)
):

    raise TypeError(
        "weight must be a number."
    )
```

It also checks whether numerical values are finite:

```python
if not math.isfinite(weight):

    raise ValueError(
        "weight must be finite."
    )
```

The same principle is used to validate spike times.

This prevents invalid values from silently propagating through the neural simulation.

---

### Dataclasses

The project uses a dataclass to organize the result of an STDP calculation.

```python
@dataclass(frozen=True)
class STDPResult:
```

The result stores:

* Presynaptic spike time.
* Postsynaptic spike time.
* Spike-time difference.
* Weight change.
* Old weight.
* New weight.

This makes the result structured and easy to access:

```python
result.delta_t

result.delta_w

result.old_weight

result.new_weight
```

---

### Mathematical Functions

The project uses Python functions to represent mathematical relationships.

The STDP model contains:

```python
def weight_change(
    delta_t_ms,
    parameters
):
```

The function determines whether the spike-time difference is positive or negative and calculates the corresponding weight change.

This demonstrates how Python can translate a mathematical neuroscience model into executable code.

---

### Abstraction

Abstraction allows the experimenter to use a high-level interface without manually controlling every internal operation.

For example:

```python
network.run()
```

causes the network to:

* Find source neurons.
* Run them.
* Detect spikes.
* Find outgoing synapses.
* Transmit spikes.
* Simulate downstream neurons.
* Queue new spike events.
* Continue propagation.

The experiment does not need to manually perform each of those operations.

The `Network` object handles the internal process.

## General Structure of the Computational Neuroscience Project

The fundamental structure of the project is:

**1. Neuron Model**

* Create a reusable neuron.
* Define membrane parameters.
* Define action-potential phases.
* Advance the neuron through time.
* Detect spikes.

**2. Synapse Model**

* Connect two neurons.
* Assign a synaptic weight.
* Transmit spikes.
* Calculate spike-time differences.
* Modify the synaptic weight through STDP.

**3. Network Model**

* Create multiple neurons.
* Create multiple synapses.
* Establish network topology.
* Propagate spikes through the network.

**4. Communication Experiments**

* Test individual neuron-to-neuron communication.
* Test synaptic transmission.
* Examine spike propagation.

**5. Network Experiments**

* Test chain networks.
* Test branching networks.
* Examine multi-hop propagation.

**6. Learning Experiments**

* Repeat network trials.
* Measure spike timing.
* Calculate STDP.
* Update synaptic weights.
* Preserve learned weights between trials.

**7. Visualization**

* Store experimental histories.
* Plot synaptic weight changes.
* Plot spike-time differences.
* Plot spike timing across trials.

The overall computational pattern is therefore:

**Neuron → Spike → Synapse → Network → Spike Timing → STDP → Weight Change → Repeated Learning → Visualization**

# B. Packages and the Python Principles Associated With Them

## NumPy

**Purpose:**

* Numerical computing.
* Numerical arrays.
* Conversion of simulation histories into numerical arrays.
* Numerical analysis of neuron voltage trajectories.

The core neuron implementation imports NumPy:

```python
import numpy as np
```

The simulation converts recorded histories into NumPy arrays:

```python
self.times = np.array(
    self.times
)

self.voltages = np.array(
    self.voltages
)
```

### Python principles associated with NumPy

* Imports
* Variables
* Arrays
* Functions
* Numerical operations
* Data structures

NumPy therefore provides the numerical data structure used to represent the simulated neural trajectory.

---

## Matplotlib

**Purpose:**

* Scientific visualization.
* Plotting learning curves.
* Visualizing synaptic weight changes.
* Visualizing spike-time differences.
* Visualizing spike timing across trials.

The visualization branch imports:

```python
import matplotlib.pyplot as plt
```

The project then plots the learning histories:

```python
plt.plot(
    trials,
    weight_12_history,
    marker="o",
    label="N1 → N2"
)
```

and:

```python
plt.plot(
    trials,
    delta_t_12_history,
    marker="o",
    label="N1 → N2"
)
```

### Python principles associated with Matplotlib

* Imports
* Functions
* Objects
* Methods
* Variables
* Lists
* Numerical data

Matplotlib converts the numerical results of the computational experiments into visual representations.

---

## math

**Purpose:**

* Mathematical calculations.
* Numerical validation.
* Exponential STDP calculations.

The project uses:

```python
import math
```

The synapse uses:

```python
math.isfinite(...)
```

to validate numerical values.

The STDP model uses:

```python
math.exp(...)
```

to calculate the exponential learning relationship.

### Python principles associated with math

* Functions
* Numerical values
* Mathematical operations
* Conditional logic
* Function inputs and outputs

The `math` module therefore provides mathematical operations used directly in the neural and plasticity models.

---

## dataclasses

**Purpose:**

* Creating structured data objects.
* Organizing the results of STDP calculations.

The project uses:

```python
from dataclasses import dataclass
```

and:

```python
@dataclass(frozen=True)
class STDPResult:
```

This creates a structured result containing:

* Pre-spike time.
* Post-spike time.
* Δt.
* Δw.
* Old weight.
* New weight.

### Python principles associated with dataclasses

* Classes
* Objects
* Attributes
* Encapsulation
* Structured data

The dataclass makes STDP results easier to store and access.

---

## Python Standard Library

The project also relies heavily on Python's built-in language features rather than requiring a large external software stack.

These include:

* Classes
* Functions
* Lists
* Dictionaries
* Loops
* Conditional statements
* Boolean logic
* Exceptions
* Object attributes
* Dataclasses
* Mathematical operations

This is important because the core neural simulation is implemented directly in Python rather than being delegated to a specialized neural-simulation framework.

## How the Packages Work Together

The packages perform different jobs within the project.

**NumPy**

→ Represents and processes numerical neural simulation data.

**math**

→ Performs mathematical calculations and numerical validation.

**dataclasses**

→ Structures STDP results into reusable Python objects.

**Matplotlib**

→ Converts experimental results into visualizations.

The Python programming principles provide the structure connecting them:

**Variables → Data Structures → Functions → Classes → Objects → Methods → Loops → Conditions → Mathematical Operations → Data Flow**

The result is a computational neuroscience system where the underlying neural mechanisms are explicitly represented in Python code rather than hidden behind a high-level neuroscience framework.
