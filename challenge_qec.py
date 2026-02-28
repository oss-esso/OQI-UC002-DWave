import json
import pennylane as qml
import pennylane.numpy as np

def circuit_left():
    """Encode logical qubit 0 into GHZ-like state across all three qubits."""
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

def circuit_right():
    """
    Coherent majority decode using only 0-1 and 1-2 connections.
    
    After encoding + single X error, state is always:
      α|s⟩ + β|s̄⟩  where s and s̄ are bitwise complements.
    
    Strategy:
      1. Majority-correct each qubit using its neighbors only
      2. Transfer logical state from qubit 0 to qubit 2
    """
    # === Step 1: Majority correction via neighbor Toffolis only ===

    # Correct qubit 1 using neighbors 0 and 2 (routed through adjacent gates)
    # "if q0=1 AND q2=1, flip q1" — but no 0-2 connection allowed!
    # Route: use q1 as intermediary to create effective Toffoli(0,2→1)
    # Decompose Toffoli(0,2→1) using only 01 and 12 connections:
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.adjoint(qml.T)(wires=2)
    qml.CNOT(wires=[0, 1])  # uses 0-1 ✓
    qml.T(wires=1)
    qml.CNOT(wires=[1, 2])  # uses 1-2 ✓
    qml.adjoint(qml.T)(wires=2)
    qml.CNOT(wires=[0, 1])  # uses 0-1 ✓
    qml.T(wires=1)
    qml.T(wires=2)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])  # uses 0-1 ✓
    qml.T(wires=0)
    qml.adjoint(qml.T)(wires=1)
    qml.CNOT(wires=[0, 1])  # uses 0-1 ✓

    # Correct qubit 0 using neighbors: only q1 is adjacent
    qml.CNOT(wires=[1, 0])  # if q1 flipped, fix q0
    # Correct qubit 2 using neighbors: only q1 is adjacent  
    qml.CNOT(wires=[1, 2])  # if q1 flipped, fix q2

    # === Step 2: Transfer logical state from qubit 0 to qubit 2 ===
    # SWAP(0,1) then SWAP(1,2) moves q0 → q2
    # SWAP via 3 CNOTs, only adjacent connections
    
    # SWAP q0 and q1
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[0, 1])
    
    # SWAP q1 and q2
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 2])

def U():
    """This operator generates a PauliX gate on a random qubit"""
    qml.PauliX(wires=np.random.randint(3))

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(alpha, beta, gamma):
    qml.U3(alpha, beta, gamma, wires=0)
    circuit_left()
    U()
    circuit_right()
    return qml.expval(0.5 * qml.PauliZ(2) - qml.PauliY(2))

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    angles = json.loads(test_case_input)
    output = circuit(*angles)
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "The expected output is not quite right."

    tape = qml.workflow.construct_tape(circuit)(2.0, 1.0, 3.0)
    ops = tape.operations

    for op in ops:
        assert not (0 in op.wires and 2 in op.wires), "Invalid connection between qubits."

    assert tape.observables[0].wires == qml.wires.Wires(2), "Measurement on wrong qubit."

# These are the public test cases
test_cases = [
    ('[2.0,1.0,3.0]', '-0.97322'),
    ('[-0.5,1.2,-1.2]', '0.88563'),
    ('[0.22,3.0,2.1]', '0.457152'),
    ('[2.22,3.1,-3.3]', '-0.335397'),
    ('[-0.2,-0.1,3.4]', '0.470199'),
    ('[-1.2,-1.1,0.4]', '-0.6494612')
]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")
    try:
        output = run(input_)
    except Exception as exc:
        print(f"Runtime Error. {exc}")
    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")
        else:
            print("Correct!")