from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

circuit = QuantumCircuit(2)



# apply hadmard to qubit 0
circuit.h(0)

# entagle qubit 0 to 1
circuit.cx(0, 1)

# measurment
circuit.measure_all()

print(circuit)

sampler = StatevectorSampler()

job = sampler.run([circuit], shots = 1000)
results = job.result()

count = results[0].data.meas.get_counts()

print(count)

