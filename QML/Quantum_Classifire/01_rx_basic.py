import pennylane as QML
from math import pi

device = QML.device("default.qubit", wires = 1)

@QML.qnode(device)
def circuit(theta):
    
    # rotwte the qubit around the 0 axis by theta
    QML.RX(theta, wires = 0 )

    # Return probability of measuring 0 or 1
    return QML.probs(wires = 0)

angels = [
    0,
    pi / 2,
    pi
]
for theta in angels:

    probability = circuit(theta)

    print("Theta:", theta)
    print("p(0): " , probability[0])
    print("p(1): " , probability[1])
    