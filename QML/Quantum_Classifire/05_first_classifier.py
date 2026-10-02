import pennylane as qml
from pennylane import numpy as np

device = qml.device("default.qubit", wires = 1)

@qml.qnode(device)
def circuit(x, theta):

    qml.RX(x, wires=0)

    qml.RY(theta, wires = 0)

    # Return probability of class 1
    return qml.probs(wires=0)

X = np.array([
    0.2,
    0.4,
    2.8,
    3.0])

# Correct class for each input
Y = np.array([
    0,
    0,
    1,
    1
])


#LOSS FUNCTION
def cost(theta):

    predictions = []

    for x in X:
        predictions.append(circuit(x, theta)[1])

    predictions = np.array(predictions)

    return np.mean((predictions - Y) ** 2)

theta = np.array(0.5, requires_grad = True)\

optimizer = qml.GradientDescentOptimizer(stepsize = 1.0)

# TRAIN

for step in range(15):
    theta = optimizer.step(cost, theta)

    print(
        "Step:", step + 1,
        "Theta:", float(theta),
        "Cost:", float(cost(theta))
    )

# Test the trained model
print("\nPredictions:")

for x, y in zip(X, Y):

    probabilities = circuit(x, theta)

    prediction = probabilities[1]
    print(
        "x:", float(x),
        "true:", int(y),
        "prediction:", float(prediction)
    )
