import pennylane as qml
from pennylane import numpy as np

device = qml.device("default.qubit", wires = 1)

@qml.qnode(device)
def circuit(x, theta1, theta2):

    #encode the input
    qml.RX(x, wires =0)

    # Trainable transformations
    qml.RY(theta1, wires=0)
    qml.RZ(theta2, wires=0)

    # Measurement
    return qml.probs(wires=0)

X = np.array([
    0.2,
    0.4,
    2.8,
    3.0
])

Y = np.array([
    0,
    0,
    1,
    1
])

def cost(parameters):
    theta1, theta2 = parameters

    predictions = []

    for x in X: 
        probabilities = circuit(x, theta1, theta2)
        predictions.append(probabilities[1])

    predictions = np.array(predictions)

    return np.mean((predictions - Y) ** 2)

parameters = np.array(
    [0.5, 0.5],
    requires_grad=True
)


optimizer = qml.GradientDescentOptimizer(
    stepsize=0.5
)


for step in range(20):

    parameters = optimizer.step(cost, parameters)

    print(
        "Step:", step + 1,
        "Theta1:", float(parameters[0]),
        "Theta2:", float(parameters[1]),
        "Cost:", float(cost(parameters))
    )