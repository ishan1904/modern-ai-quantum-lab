# First classifiers: input, prediction, error, update

## What happens in `05_first_classifier.py`?

1. Start each circuit call with a fresh qubit in state |0>.
2. `RX(x)` puts this sample's input angle into the state. The four inputs stay fixed.
3. `RY(theta)` applies the adjustable rotation. Every sample uses the same theta.
4. `qml.probs(wires=0)` returns [P(0), P(1)]. Index 1 is the prediction for class 1; the circuit itself returns both probabilities.
5. Run all four samples, subtract their correct labels, square each error, and average. This is mean-squared error (MSE).
6. PennyLane differentiates that loss with respect to theta. Gradient descent updates `theta = theta - learning_rate * gradient`.
7. Repeat 15 times, with learning rate 1.0 and theta initially 0.5. The printed loss is calculated after each update.

For example, a class-1 sample with P(1)=0.8 contributes (0.8-1)^2=0.04 to the loss. Input x says **which sample**; theta controls **how the model processes samples**. Only theta is learned. The last loop predicts the training samples again, so it does not measure performance on unseen inputs.

This simple architecture has P(1) = (1 - cos(x) cos(theta))/2. RX already separates inputs near 0 from inputs near pi. Training tunes that separation; it does not learn a complicated boundary from scratch.

## Next experiment: `06_two_qubit_classifier.py`

The functions separate dataset handling (`load_dataset`), circuit design, prediction, loss, training, and evaluation. Importing the module does not run training.

The baseline reproduces the original one-qubit architecture. The two-qubit circuit encodes the same scalar as RX(x) on wire 0 and RX(x/2) on wire 1, applies one trainable RY on each wire, applies CNOT(0 -> 1), then applies a third trainable RY on the readout wire. CNOT is the entangling layer: it can create correlations and entanglement for suitable inputs. It does not guarantee every state is entangled. Measuring wire 1 returns its marginal [P(0), P(1)], not the four joint probabilities.

The half-angle encoding avoids making both wires identical: identical encoding with this readout can erase the useful distinction between angles near 0 and pi. It is a fixed design choice, not another learned parameter. The final RY lets training adjust the readout basis. This is a deliberately small ansatz; parameter count does not guarantee independent expressive capacity.

Both models use the same original four training samples, 15 full-batch gradient updates, learning rate 1.0, MSE, and initial angle 0.5 for each parameter. Exact `default.qubit` probabilities remove shot noise. Six distinct test inputs use the illustrative rule class 0 below pi/2 and class 1 above pi/2 on [0, pi]. Test labels never enter training. A probability >=0.5 predicts class 1.

From the repository root, in your activated virtual environment:

```bash
python -m pip install pennylane==0.45.1
python QML/Quantum_Classifire/06_two_qubit_classifier.py
python -m unittest discover -s QML/Quantum_Classifire -p 'test_*.py'
```

The script prints circuit drawings, initial and learned parameters, loss histories, training/test MSE and accuracy, and individual test predictions. A verified run with PennyLane 0.45.1 produced:

| Model | Parameters | Initial train MSE | Final train MSE | Test MSE | Test accuracy |
|---|---:|---:|---:|---:|---:|
| One qubit | 1 | 0.006469 | 0.001399 | 0.023168 | 100% |
| Two qubits | 3 | 0.144177 | 0.099926 | 0.103601 | 100% |

The one-qubit model has lower error here; the two-qubit model makes less confident class-1 predictions. This comparison changes architecture, encoding, and parameter count together, so it cannot isolate an entanglement benefit. Four training and six test samples demonstrate the workflow, not reliable real-world generalization or quantum advantage. Try changing the initialization or training budget to see their effects, keeping both models' budgets matched.
