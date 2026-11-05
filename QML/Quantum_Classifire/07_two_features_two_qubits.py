import pennylane as qml 
from pennylane import numpy as np

device = qml.device("default.qubit", wires = 2)

@qml.qnode(device)
def circuit(x):

    # Encode feature 1 into qubit 0
    qml.RX(x[0], wires = 0)

    # Encode feature 2 into qubit 1
    qml.RX(x[1], wires=1)

    return qml.probs(wires = [0,1])

x = np.array([0.7, 1.2])

probabilities = circuit(x)

print("Input:", x)
print("Probabilities:", probabilities)