"""Check the gradient path, baseline equivalence, and evaluation contract."""

import importlib.util
from pathlib import Path
import unittest

import pennylane as qml
from pennylane import numpy as np

spec = importlib.util.spec_from_file_location(
    "classifier", Path(__file__).with_name("06_two_qubit_classifier.py")
)
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)


class ClassifierTests(unittest.TestCase):
    def test_baseline_matches_analytic_probability(self):
        for x in [0.2, 1.0, 3.0]:
            for theta in [0.0, 0.5, 1.2]:
                probability = classifier.one_qubit_circuit(x, np.array([theta]))[1]
                self.assertAlmostEqual(float(probability),
                                       float((1 - np.cos(x) * np.cos(theta)) / 2))

    def test_two_qubit_gradients_and_training(self):
        x, y, _, _ = classifier.load_dataset()
        parameters = np.array([0.3, 0.6, -0.2], requires_grad=True)
        loss = lambda p: classifier.mean_squared_error(
            classifier.two_qubit_circuit, p, x, y
        )
        gradient = qml.grad(loss)(parameters)
        for index in range(3):
            direction = np.zeros(3, requires_grad=False)
            direction[index] = 1e-6
            finite_difference = (loss(parameters + direction) - loss(parameters - direction)) / 2e-6
            self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=6)
            self.assertGreater(abs(float(gradient[index])), 1e-6)
        trained, history = classifier.train(classifier.two_qubit_circuit, [0.5] * 3, x, y)
        self.assertEqual(len(history), 16)
        self.assertLess(history[-1], history[0])
        self.assertFalse(np.allclose(trained, [0.5] * 3))

    def test_dataset_and_evaluation(self):
        x, y, test_x, test_y = classifier.load_dataset()
        self.assertFalse(set(map(float, x)) & set(map(float, test_x)))
        self.assertFalse(x.requires_grad)
        for circuit, initial in [(classifier.one_qubit_circuit, [0.5]),
                                 (classifier.two_qubit_circuit, [0.5] * 3)]:
            parameters, _ = classifier.train(circuit, initial, x, y)
            result = classifier.evaluate(circuit, parameters, test_x, test_y)
            self.assertTrue(np.all((result['probabilities'] >= 0) &
                                   (result['probabilities'] <= 1)))
            self.assertEqual(result['probabilities'].shape, test_y.shape)
            self.assertEqual(result['accuracy'], 1.0)
            self.assertTrue(np.array_equal(result['labels'], test_y))


if __name__ == "__main__":
    unittest.main()
