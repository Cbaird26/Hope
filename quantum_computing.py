from qiskit import IBMQ, QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import streamlit as st

def initialize_ibmq():
    api_key = st.text_input("Enter your IBMQ API Key:", type="password")
    if st.button("Load IBMQ Account"):
        IBMQ.save_account(api_key)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        st.success("IBMQ account loaded successfully!")
        return provider
    else:
        st.warning("Please enter your IBMQ API Key and click 'Load IBMQ Account'.")
        return None

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
