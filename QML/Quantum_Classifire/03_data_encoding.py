import pennylane as qml
from pennylane import numpy as np

# One simulated qubit
device = qml.device("default.qubit", wires=1)

@qml.qnode(device)
def encode_data(x):

     # Encode classical value x into the qubit
    qml.RX(x, wires = 0)

    # probability of measuring |0> or |1>
    return qml.probs(wires = 0)

data_points = [
     0.0,
    np.pi / 4,
    np.pi / 2,
    np.pi
]

for x in data_points:
    probabilities = encode_data(x)

    print("Input x:", float(x))
    print("P(0):", float(probabilities[0]))
    print("P(1):", float(probabilities[1]))
    print()