"""Compare the original RX/RY classifier with a small entangling circuit.

Run: python QML/Quantum_Classifire/06_two_qubit_classifier.py
Dependencies: pennylane (which includes its NumPy interface).
"""

import pennylane as qml
from pennylane import numpy as np


def load_dataset():
    """Angles in radians; fixed labels are never optimized.

    The toy rule is class 0 below pi/2 and class 1 above it on [0, pi].
    Test angles are distinct from training angles; this is a tiny demo,
    not evidence of generalization on a real dataset.
    """
    train_x = np.array([0.2, 0.4, 2.8, 3.0], requires_grad=False)
    train_y = np.array([0, 0, 1, 1], requires_grad=False)
    test_x = np.array([0.1, 0.6, 1.0, 2.1, 2.6, 3.1], requires_grad=False)
    test_y = np.array([0, 0, 0, 1, 1, 1], requires_grad=False)
    return train_x, train_y, test_x, test_y


@qml.qnode(qml.device("default.qubit", wires=1), interface="autograd")
def one_qubit_circuit(x, parameters):
    """Same architecture as 05_first_classifier.py."""
    qml.RX(x, wires=0)
    qml.RY(parameters[0], wires=0)
    return qml.probs(wires=0)


@qml.qnode(qml.device("default.qubit", wires=2), interface="autograd")
def two_qubit_circuit(x, parameters):
    """Encode the same scalar at two scales; no extra feature is introduced."""
    # Input encoding: x describes this sample, not a learned parameter.
    qml.RX(x, wires=0)
    qml.RX(x / 2, wires=1)
    # Two trainable rotations before the entangling layer.
    qml.RY(parameters[0], wires=0)
    qml.RY(parameters[1], wires=1)
    qml.CNOT(wires=[0, 1])
    # Rotate the readout basis so correlations can affect the prediction.
    qml.RY(parameters[2], wires=1)
    # Marginal probabilities [P(0), P(1)] of the readout qubit.
    return qml.probs(wires=1)


def predict_probabilities(circuit, parameters, inputs):
    """Extract P(class 1) while preserving the gradient path."""
    return np.stack([circuit(x, parameters)[1] for x in inputs])


def mean_squared_error(circuit, parameters, inputs, labels):
    predictions = predict_probabilities(circuit, parameters, inputs)
    return np.mean((predictions - labels) ** 2)


def train(circuit, initial_parameters, inputs, labels, steps=15, learning_rate=1.0):
    """Only the parameter vector is passed to the optimizer."""
    parameters = np.array(initial_parameters, requires_grad=True)
    optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)
    history = [float(mean_squared_error(circuit, parameters, inputs, labels))]

    def loss(current_parameters):
        return mean_squared_error(circuit, current_parameters, inputs, labels)

    for _ in range(steps):
        parameters = optimizer.step(loss, parameters)
        history.append(float(loss(parameters)))
    return parameters, history


def evaluate(circuit, parameters, inputs, labels):
    probabilities = predict_probabilities(circuit, parameters, inputs)
    predicted_labels = (probabilities >= 0.5).astype(int)
    return {
        "mse": float(np.mean((probabilities - labels) ** 2)),
        "accuracy": float(np.mean(predicted_labels == labels)),
        "probabilities": probabilities,
        "labels": predicted_labels,
    }


def main():
    train_x, train_y, test_x, test_y = load_dataset()
    models = [
        ("1 qubit", one_qubit_circuit, [0.5]),
        ("2 qubits", two_qubit_circuit, [0.5, 0.5, 0.5]),
    ]
    print("Same data, MSE, 15 full-batch steps, learning rate 1.0, exact simulation.")
    print("Different architectures and parameter counts; no quantum advantage claim.\n")
    print("Model     Params  Initial MSE  Train MSE  Train acc  Test MSE   Test acc")
    for name, circuit, initial_parameters in models:
        parameters, history = train(circuit, initial_parameters, train_x, train_y)
        training = evaluate(circuit, parameters, train_x, train_y)
        testing = evaluate(circuit, parameters, test_x, test_y)
        print(f"{name:<9} {len(parameters):>6}  {history[0]:>11.6f}  "
              f"{training['mse']:>9.6f}  {training['accuracy']:>9.1%}  "
              f"{testing['mse']:>8.6f}  {testing['accuracy']:>9.1%}")
        print("  Initial parameters:", initial_parameters)
        print("  Learned parameters:", parameters)
        print("  Loss history (initial, then each update):")
        print(" ", [round(value, 6) for value in history])
        print("  Circuit:\n", qml.draw(circuit)(test_x[0], parameters))
        print("  Unseen inputs: x, true label, P(1), predicted label")
        for x, y, probability, label in zip(
            test_x, test_y, testing["probabilities"], testing["labels"]
        ):
            print(f"    {float(x):.2f}  {int(y)}  {float(probability):.6f}  {int(label)}")
        print()


if __name__ == "__main__":
    main()
