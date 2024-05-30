from qiskit import IBMQ, QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

def initialize_ibmq():
    # Load your IBM Q account
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    return provider

def create_quantum_circuit():
    # Create a simple quantum circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def extract_features(qc):
    # Example feature extraction from quantum circuit
    simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, simulator)
    qobj = assemble(transpiled_qc)
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    features = list(counts.values())
    return features
