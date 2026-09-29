import pennylane as qml
from pennylane import numpy as np

# One simulated qubit
device = qml.device("default.qubit", wires=1)

@qml.qnode(device)
def circuit(theta):
    qml.RX(theta, wires = 0)

    # probability of measuring |0> or |1>
    return qml.probs(wires = 0)

# We want the circuit to produce |1>
def cost(theta):

    probabilities = circuit(theta)

    p_1 = probabilities[1]

    # if p_1(1) = 1 cost = 0
    return 1 - p_1

theta = np.array(0.1 , requires_grad = True )

# gradient descent optimizer
optimizer = qml.GradientDescentOptimizer(stepsize= 1.0)

for step in range(15):

    theta = optimizer.step(cost, theta)
    probabilities = circuit(theta)
    

    print(
        "Step:", step + 1,
        "Theta:", float(theta),
        "P(1):", float(probabilities[1]),
        "Cost:", float(cost(theta))
    )
