import pennylane as qml
from pennylane import numpy as np

device = qml.device("default_qubit", wires = 1)

@qml.qnode(device)
def circuit(x, theta):
  
    # x = input data
    qml.RX(x, wires = 0)

    # theta = trainable model parameter
    qml.RY(theta, wires = 0)

    return qml.probs(wires=0)

theta = 0.5

data_points = [
    0.0,
    np.pi / 4,
    np.pi / 2,
    np.pi
]

for x in data_points:
    probabilities = circuit(x, theta)

    print("Input x:", float(x))
    print("Theta:", theta)
    print("P(0):", float(probabilities[0]))
    print("P(1):", float(probabilities[1]))
    print()