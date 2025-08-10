#%%
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')  # or 'QtAgg', depending on what's installed
#%%
# Build the Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Get simulator backend
simulator = Aer.get_backend('aer_simulator')

# Transpile the circuit for the backend
compiled = transpile(qc, simulator)

# Run the circuit on the simulator
job = simulator.run(compiled, shots=1024)
result = job.result()

# Get and plot the result
counts = result.get_counts()
print(counts)  # This should show something like {'00': 500, '11': 524}


# plot_histogram(counts)

# Extract labels and values
labels = list(counts.keys())
values = list(counts.values())

# Plot manually using bar chart
plt.bar(labels, values, color='skyblue', edgecolor='black')
plt.xlabel('Measurement outcome')
plt.ylabel('Counts')
plt.title('Manual Histogram of Bell State Results')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
#%%
qc.draw('mpl')


# %%

# Step 1: Create circuit with 10 qubits and 10 classical bits
qc = QuantumCircuit(10, 10)

# Step 2: Create 5 Bell pairs (0-1, 2-3, 4-5, 6-7, 8-9)
for i in range(0, 10, 2):
    qc.h(i)         # Put qubit i in superposition
    qc.cx(i, i + 1) # Entangle with qubit i+1

# Step 3: Measure all qubits
qc.measure(range(10), range(10))

# Step 4: Run the simulation
sim = Aer.get_backend('aer_simulator')
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=1024).result()
counts = result.get_counts()
#%%
# Step 5: Plot histogram
plot_histogram(counts, figsize=(12, 5))
# plt.title("Measurement Outcomes of 10-Qubit Bell-Pair Circuit")
# plt.show()
#%%
# Optional: print a few outcomes
print("Sample outcomes:")
for outcome, count in list(counts.items())[:5]:
    print(f"{outcome}: {count}")

# %%

# This creates 2 entangled states of 3 qubits
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()



# Get simulator backend
simulator = Aer.get_backend('aer_simulator')

# Transpile the circuit for the backend
compiled = transpile(qc, simulator)

# Run the circuit on the simulator
job = simulator.run(compiled, shots=1024)
result = job.result()

# Get and plot the result
counts = result.get_counts()
print(counts)  # This should show something like {'00': 500, '11': 524}


# plot_histogram(counts)

# Extract labels and values
labels = list(counts.keys())
values = list(counts.values())

# Plot manually using bar chart
plt.bar(labels, values, color='skyblue', edgecolor='black')
plt.xlabel('Measurement outcome')
plt.ylabel('Counts')
plt.title('Manual Histogram of Bell State Results')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
#%%

#This makes 8 states of superposition between the 3 states
qc = QuantumCircuit(3, 3)

# Put all 3 qubits into independent superposition
qc.h(0)
qc.h(1)
qc.h(2)

# Measure all
qc.measure([0,1,2], [0,1,2])


# Get simulator backend
simulator = Aer.get_backend('aer_simulator')

# Transpile the circuit for the backend
compiled = transpile(qc, simulator)

# Run the circuit on the simulator
job = simulator.run(compiled, shots=1024)
result = job.result()

# Get and plot the result
counts = result.get_counts()
print(counts)  # This should show something like {'00': 500, '11': 524}


# plot_histogram(counts)

# Extract labels and values
labels = list(counts.keys())
values = list(counts.values())

# Plot manually using bar chart
plt.bar(labels, values, color='skyblue', edgecolor='black')
plt.xlabel('Measurement outcome')
plt.ylabel('Counts')
plt.title('Manual Histogram of Bell State Results')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
#%%

qc = QuantumCircuit(5)
qc.h(0); qc.cx(0, 1)
qc.h(2); qc.cx(2, 3)
qc.h(4)  # just superpose the last one
qc.measure_all()


# Get simulator backend
simulator = Aer.get_backend('aer_simulator')

# Transpile the circuit for the backend
compiled = transpile(qc, simulator)

# Run the circuit on the simulator
job = simulator.run(compiled, shots=1024)
result = job.result()

# Get and plot the result
counts = result.get_counts()
print(counts)  # This should show something like {'00': 500, '11': 524}


# plot_histogram(counts)

# Extract labels and values
labels = list(counts.keys())
values = list(counts.values())

# Plot manually using bar chart
plt.bar(labels, values, color='skyblue', edgecolor='black')
plt.xlabel('Measurement outcome')
plt.ylabel('Counts')
plt.title('Manual Histogram of Bell State Results')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
# %%

# Create circuit: 3 qubits and 1 classical bit for result
qc = QuantumCircuit(3, 1)

# Set inputs A=1, B=1 (use X gates to flip from 0 to 1)
qc.x(0)  # A = 1
qc.x(1)  # B = 1

# Toffoli gate with controls q0 and q1, target q2
qc.ccx(0, 1, 2)

# Measure the result (stored in q2)
qc.measure(2, 0) #and display in classical bit 0 (the only one here for now)

# Run the circuit
# Get backend (simulator)
simulator = Aer.get_backend('aer_simulator')

# Transpile circuit for backend
compiled = transpile(qc, simulator)

# Run
job = simulator.run(compiled, shots=1024)
result = job.result()

counts = result.get_counts()

# Show result
plot_histogram(counts)
# plt.show()

# %%
from qiskit import QuantumCircuit
import numpy as np

qc = QuantumCircuit(1,1)
qc.rx(np.pi/3, 0)  # rotate around X axis
qc.measure_all()
qc.draw('mpl')

# %%
qc = QuantumCircuit(2,2)
qc.swap(0,1)
qc.measure_all()

# %%
from qiskit.quantum_info import Statevector
import qiskit
qc = QuantumCircuit(1)
qc.h(0)
sv = Statevector.from_instruction(qc)
print(sv)
qiskit.visualization.plot_bloch_multivector(sv)

# %%
