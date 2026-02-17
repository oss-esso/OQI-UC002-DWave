# Microsoft Research Intern Interview Preparation
## Position CS-632: Research Intern — Quantum Computing (Chemistry & Physics Applications)

**Prepared for:** Edoardo Spigarolo, M.Sc. Quantum Science & Engineering, EPFL  
**Date:** January 2026  

---

## Table of Contents
1. [Task 1: MSR Quantum Interview Format & Questions](#task-1)
2. [Task 2: VQE Crash Course](#task-2)
3. [Task 3: QEC & Surface Codes Crash Course](#task-3)
4. [Task 4: Computational Chemistry on Quantum Computers](#task-4)
5. [Task 5: Circuit-Based QC Fundamentals](#task-5)
6. [Task 6: Microsoft Quantum Stack Deep Dive](#task-6)
7. [Task 7: 7-Day Study Plan](#task-7)

---

<a name="task-1"></a>
## Task 1: MSR Quantum Interview Format & Likely Questions

### 1.1 Interview Format (based on research)

Microsoft Research internship interviews for quantum computing typically involve:

1. **Initial Screen (30-60 min):** Phone/video call with the hiring manager or a team member. Expect questions about your background, research motivation, and high-level quantum knowledge.
2. **Technical Interview Round(s) (1-3 sessions, 45-60 min each):**
   - Whiteboard/coding problems in quantum algorithms or physics
   - Deep-dive into your past research (D-Wave annealing, transmon simulation)
   - Problem-solving exercises: designing quantum circuits, analyzing error models
3. **Research Presentation (optional for some positions):** You may be asked to present your thesis work or a research topic.
4. **Behavioral/Culture Fit:** Growth mindset, collaboration, intellectual curiosity.

### 1.2 Active Intern Postings (as of Jan 2026)

From Microsoft Research's quantum computing page:
- **Research Intern — Quantum Applications** (posting date: Jan 28, 2026)
- **Research Intern — Quantum Algorithms** (posting date: Jan 27, 2026)
- **Research Intern — Quantum Error Correction** (posting date: Dec 18, 2025)

### 1.3 Likely Interview Questions for CS-632

**Fundamentals:**
- What is a qubit? Explain superposition and entanglement.
- Describe the Bloch sphere representation.
- What is the difference between gate-based and adiabatic/annealing quantum computing?
- Explain the no-cloning theorem and its implications.
- What are T1 and T2 relaxation times? How do they affect computation?

**VQE & Quantum Chemistry (HIGH PRIORITY for this role):**
- Explain VQE. What is the variational principle?
- What is a parameterized quantum circuit (ansatz)? Describe UCCSD.
- How does Jordan-Wigner (or Bravyi-Kitaev) transformation work?
- What is the difference between VQE and QPE for chemistry simulation?
- How would you choose the active space for a molecular simulation?
- What are the main challenges in scaling VQE to larger molecules?
- Explain barren plateaus and how they affect VQE optimization.

**Quantum Error Correction (HIGH PRIORITY):**
- What is a stabilizer code? Explain the surface code.
- What is a logical qubit vs. a physical qubit?
- Explain the threshold theorem for fault-tolerant quantum computation.
- What is lattice surgery? How do you perform logical gates on surface codes?
- What is magic state distillation and why is it needed?
- What is the difference between error detection and error correction?

**Microsoft-Specific (KNOW THIS):**
- What is a topological qubit? How does it differ from superconducting/trapped-ion qubits?
- Explain Majorana Zero Modes and non-Abelian anyons.
- What is a topoconductor?
- What is the Majorana 1 chip? What milestone does it represent?
- What is the Azure Quantum Resource Estimator and why is it important?
- Describe Microsoft's quantum computing roadmap (Level 1 → 2 → 3).

**Your Background (be ready to bridge to the role):**
- How does your D-Wave quantum annealing experience translate to gate-based QC?
- What optimization problems did you solve? How might they map to VQE?
- Describe your transmon qubit simulation work. What physics did you model?
- How would you approach simulating a molecule's ground state energy?

### 1.4 Key Resources for Interview Prep

| Resource | URL | Priority |
|----------|-----|----------|
| Microsoft Research Quantum Computing | https://www.microsoft.com/research/research-area/quantum-computing/ | ⭐⭐⭐ |
| Microsoft Quantum Roadmap | https://quantum.microsoft.com/en-us/our-story | ⭐⭐⭐ |
| Azure Quantum Overview | https://learn.microsoft.com/en-us/azure/quantum/overview-azure-quantum | ⭐⭐⭐ |
| Microsoft Quantum Blog | https://azure.microsoft.com/en-us/blog/quantum/ | ⭐⭐⭐ |
| Quantum Katas (self-paced) | https://quantum.microsoft.com/en-us/tools/quantum-katas | ⭐⭐ |

---

<a name="task-2"></a>
## Task 2: VQE (Variational Quantum Eigensolver) Crash Course

### 2.1 Core Concept

VQE uses the **variational principle** to find the ground state energy of a quantum system:

$$E_0 \leq \langle \psi(\theta) | H | \psi(\theta) \rangle$$

The algorithm:
1. **Prepare** a parameterized trial state $|\psi(\theta)\rangle$ on a quantum computer
2. **Measure** the expectation value $\langle H \rangle = \langle \psi(\theta)|H|\psi(\theta)\rangle$
3. **Optimize** parameters $\theta$ classically to minimize $\langle H \rangle$
4. **Repeat** until convergence

### 2.2 Key Components

#### Hamiltonian Encoding
- **Second quantization:** Express molecular Hamiltonian using creation/annihilation operators
  $$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$$
- **Jordan-Wigner transformation:** Maps fermionic operators to Pauli operators
  - $a_j^\dagger \rightarrow \frac{1}{2}(X_j - iY_j) \otimes Z_{j-1} \otimes \cdots \otimes Z_0$
  - Preserves anti-commutation relations; requires O(N) Pauli weight
- **Bravyi-Kitaev transformation:** Alternative mapping with O(log N) Pauli weight
- Result: Hamiltonian as weighted sum of Pauli strings: $H = \sum_i c_i P_i$

#### Ansatz (Trial Wavefunction)
- **Hardware-efficient ansatz:** Layers of parameterized single-qubit rotations + entangling gates. Easy to implement but may suffer from barren plateaus.
- **UCCSD (Unitary Coupled Cluster Singles and Doubles):**
  $$|\psi\rangle = e^{T - T^\dagger}|HF\rangle$$
  where $T = T_1 + T_2$ includes single and double excitation operators. Chemically motivated; good accuracy but deep circuits.
- **Adaptive VQE (ADAPT-VQE):** Grows the ansatz operator-by-operator based on gradient magnitudes.

#### Measurement
- Measure each Pauli string $P_i$ in the Hamiltonian decomposition
- Group commuting terms to reduce measurement overhead
- **Classical shadows:** Efficient method to estimate many expectation values from fewer measurements (used in MS+Quantinuum chemistry demo)

#### Classical Optimization
- Gradient-based: Adam, L-BFGS, natural gradient
- Gradient-free: COBYLA, Nelder-Mead, SPSA
- Parameter-shift rule enables exact gradient computation on quantum hardware

### 2.3 Practical VQE Code (PennyLane)

From the PennyLane VQE tutorial (https://pennylane.ai/qml/demos/tutorial_vqe/):

```python
import pennylane as qml
from pennylane import numpy as np

# Load molecular data for H2
dataset = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=0.742)[0]
H = dataset.hamiltonian
hf_state = dataset.hf_state  # e.g., [1, 1, 0, 0]
qubits = len(hf_state)

dev = qml.device("default.qubit", wires=qubits)

@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))
    # Double excitation: |1100⟩ → |0011⟩
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    return qml.expval(H)

# Optimize
import optax
opt = optax.sgd(learning_rate=0.4)
params = np.array([0.0])

for step in range(100):
    grads = qml.grad(circuit)(params)
    updates, opt_state = opt.update(grads, opt.init(params))
    params = optax.apply_updates(params, updates)
    if step % 20 == 0:
        print(f"Step {step}: Energy = {circuit(params):.8f} Ha")
# Converges to ≈ -1.13726 Ha (exact: -1.1373 Ha)
```

### 2.4 VQE vs QPE Comparison

| Feature | VQE | QPE |
|---------|-----|-----|
| Circuit depth | Shallow | Deep |
| Error tolerance | NISQ-friendly | Requires fault tolerance |
| Accuracy | Limited by ansatz & optimization | Exponentially precise |
| Classical cost | High (many measurements + optimization) | Low post-processing |
| Best for | Near-term devices | Future fault-tolerant QC |

### 2.5 Key References & Resources

| Resource | URL | Type |
|----------|-----|------|
| **PennyLane VQE Tutorial** | https://pennylane.ai/qml/demos/tutorial_vqe/ | Interactive tutorial with code |
| **PennyLane Molecular Hamiltonians** | https://pennylane.ai/qml/demos/tutorial_quantum_chemistry/ | Building Hamiltonians tutorial |
| **PennyLane Chemistry Docs** | https://docs.pennylane.ai/en/stable/introduction/chemistry.html | API reference |
| **VQE Review Paper (Tilly et al. 2022)** | https://arxiv.org/abs/2111.05176 | Comprehensive 156-page review |
| **Original VQE Paper (Peruzzo et al. 2014)** | https://arxiv.org/abs/1304.3061 | Foundation paper, Nature Comm. 5:4213 |
| **PennyLane Codebook: Variational Quantum Algorithms** | https://pennylane.ai/codebook/variational-quantum-algorithms | Self-paced exercises |

**YouTube Recommendations:**
- "Variational Quantum Eigensolver (VQE) - Qiskit Summer School" — IBM Qiskit YouTube channel
- "Quantum Chemistry with VQE" — PennyLane YouTube channel
- "Introduction to VQE" — Xanadu (PennyLane) YouTube (@pennylaneai)

---

<a name="task-3"></a>
## Task 3: Quantum Error Correction & Surface Codes Crash Course

### 3.1 Why QEC?

Quantum computers are inherently noisy. Physical qubits have error rates ~0.1-1%. For useful computation, we need logical error rates ~$10^{-10}$ or better. QEC encodes logical qubits into many physical qubits, enabling detection and correction of errors without measuring (and collapsing) the encoded quantum state.

### 3.2 Stabilizer Formalism

**Key idea:** Define a code space as the +1 eigenspace of a set of commuting Pauli operators called **stabilizers**.

For an $[[n, k, d]]$ code:
- $n$ = number of physical qubits
- $k$ = number of logical qubits
- $d$ = code distance (minimum weight of undetectable error)

**Syndrome measurement:** Measure each stabilizer to detect errors. The pattern of ±1 outcomes (syndrome) identifies the error without revealing the encoded quantum state.

### 3.3 The Surface Code

The surface code is the leading candidate for fault-tolerant quantum computation:

**Structure:**
- Qubits placed on edges of a 2D lattice
- **X-type stabilizers** (stars/vertices): Products of X operators on edges around a vertex
- **Z-type stabilizers** (plaquettes/faces): Products of Z operators on edges around a face
- $[[L^2 + (L-1)^2, 1, L]]$ code for a planar patch with distance $L$

**Key properties:**
- Only nearest-neighbor interactions required (great for superconducting qubits)
- **Threshold:** ~1% circuit-level error rate (achievable by current hardware!)
  - 1.8% under correlated CNOT errors
  - 0.57% for depolarizing noise with MWPM decoding
- A logical qubit with $10^{-14}$ error rate needs ~$10^3$–$10^4$ physical qubits

**Error correction procedure:**
1. Measure all stabilizers (syndrome extraction)
2. Feed syndrome to a classical **decoder** (e.g., MWPM, Union-Find)
3. Apply corrections based on decoder output
4. Repeat each QEC round

**Decoders (important topic):**
- **Minimum Weight Perfect Matching (MWPM):** Maps error correction to graph matching problem. Near-optimal but O(n³).
- **Union-Find:** Nearly linear time, hardware-friendly
- **Neural network decoders:** Trained on error patterns (e.g., Google's AlphaQubit)

### 3.4 Logical Operations on Surface Codes

| Operation | Method | Complexity |
|-----------|--------|------------|
| Pauli X, Z | Transversal | Easy |
| CNOT | Lattice surgery | Moderate |
| Hadamard | Code deformation | Moderate |
| S gate | Code deformation | Moderate |
| T gate | Magic state distillation | Expensive |

**Lattice surgery:** Merge and split surface code patches to perform logical CNOT operations. Key technique for practical computation.

**Magic state distillation:** Non-Clifford gates (like T) cannot be performed transversally. Instead:
1. Prepare noisy "magic states" $|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$
2. Distill them to high fidelity using Clifford operations
3. Consume distilled states to implement T gates via gate teleportation

This is the dominant overhead in fault-tolerant quantum computing!

### 3.5 Beyond Surface Codes

- **Color codes:** Higher-rate alternatives, transversal Clifford gates
- **LDPC codes:** Constant rate, less overhead, but non-local connectivity required
- **Floquet codes:** Error correction through periodic measurements (honeycomb code)
- **Microsoft's 4D codes (Jun 2025):** Novel family reducing overhead vs. surface codes
- **Topological codes (Microsoft):** Hardware-protected qubits + custom Floquet codes reduce overhead ~10x compared to prior approaches

### 3.6 Key References & Resources

| Resource | URL | Type |
|----------|-----|------|
| **Fowler et al. "Surface codes: Towards practical large-scale QC" (2012)** | https://arxiv.org/abs/1208.0928 | Essential 54-page introduction |
| **Error Correction Zoo: Surface Code** | https://errorcorrectionzoo.org/c/surface | Comprehensive reference with all variants |
| **PennyLane Codebook: QEC Module** | https://pennylane.ai/codebook/quantum-error-correction | Interactive exercises |
| **Quantum Katas: QEC (Bit Flip, Phase Flip, Shor Codes)** | https://quantum.microsoft.com/en-us/tools/quantum-katas | Microsoft's self-paced tutorials |
| **Microsoft 4D Codes Blog** | https://azure.microsoft.com/en-us/blog/quantum/2025/06/19/microsoft-advances-quantum-error-correction-with-a-family-of-novel-four-dimensional-codes/ | Latest MS QEC research |

**YouTube Recommendations:**
- "Quantum Error Correction" — MIT OpenCourseWare (Peter Shor)
- "Surface Codes" — Talks at Google (Austin Fowler)
- "Introduction to Topological Quantum Error Correction" — Perimeter Institute (recorded lectures)
- "QEC: From Theory to Practice" series — Qiskit YouTube channel

---

<a name="task-4"></a>
## Task 4: Computational Chemistry on Quantum Computers

### 4.1 Why Quantum Chemistry on QC?

Classical computers struggle with **strongly correlated** electronic systems where electrons are heavily entangled. The Hilbert space grows exponentially: $N$ spin-orbitals → $2^N$ basis states. Quantum computers can represent these states natively.

**Key applications:**
- Drug discovery (molecular binding energies)
- Catalyst design (nitrogen fixation — MS's nitrogenase demo)
- Materials science (superconductors, batteries)
- Carbon capture chemistry

### 4.2 The Computational Pipeline

```
Molecule → Classical Pre-processing → Quantum Algorithm → Post-processing
           (Hartree-Fock, basis set,    (VQE or QPE)     (energy extraction,
            active space selection)                        property calculation)
```

#### Step 1: Classical Pre-processing
- **Basis set selection:** STO-3G (minimal), cc-pVDZ (double-zeta), etc.
- **Hartree-Fock (HF):** Single-determinant approximation, gives reference state
- **Active space selection:** Identify chemically important orbitals
  - Core orbitals: always doubly occupied
  - Active orbitals: variable occupation (simulated on quantum computer)
  - Virtual orbitals: always empty
- **Number of qubits = 2 × (number of active orbitals)** (factor of 2 for spin)

#### Step 2: Fermion-to-Qubit Mapping
- **Jordan-Wigner:** Simple, O(N) Pauli weight, preserves locality of occupation numbers
- **Bravyi-Kitaev:** O(log N) Pauli weight, faster Hamiltonian simulation
- **Parity mapping:** Encodes parity of occupation numbers
- **Qubit tapering:** Exploits molecular symmetries (e.g., electron number, spin) to reduce qubit count

#### Step 3: Quantum Algorithm
| Algorithm | Circuit Depth | Accuracy | Hardware Requirements |
|-----------|--------------|----------|----------------------|
| VQE | Shallow | Depends on ansatz | NISQ |
| QPE | Deep (~$1/\epsilon$) | Arbitrary precision | Fault-tolerant |
| UCCSD-VQE | Moderate | Chemical accuracy | Near-term |
| Quantum subspace expansion | Shallow | Iteratively improved | NISQ |

#### Step 4: Post-processing
- Extract energy eigenvalues
- Compute molecular properties (dipole moments, forces)
- Error mitigation (zero-noise extrapolation, probabilistic error cancellation)

### 4.3 Microsoft's Chemistry Demonstration (Sep 2024)

**Landmark result:** Microsoft + Quantinuum demonstrated end-to-end hybrid chemistry simulation:
1. **HPC (Azure):** AutoCAS + AutoRXN identified active space and reaction pathways of PNNP iron catalyst
2. **Quantum (Quantinuum H1):** 2 logical qubits prepared ground state of active space
3. **AI:** Classical shadows used to train AI model on quantum properties
4. **Result:** Ground state energy estimated within chemical accuracy (1.6 mHa)
5. **Key finding:** Logical qubits gave better estimates than physical qubits with 97% probability

This is the **first demonstration of HPC + quantum + AI** solving a chemistry problem together.

### 4.4 Key Concepts to Know

- **Chemical accuracy:** 1 kcal/mol ≈ 1.6 mHa (milliHartree) — the precision needed for chemically meaningful predictions
- **Correlation energy:** The energy difference between Hartree-Fock and exact solutions; captures electron-electron interactions
- **Active space:** The subset of orbitals where electron correlation is significant
- **Classical shadows:** Efficient protocol to extract multiple expectation values from randomized measurements

### 4.5 Tools & Software

| Tool | Description | URL |
|------|-------------|-----|
| **PennyLane qchem** | Full pipeline: molecule → Hamiltonian → VQE | https://docs.pennylane.ai/en/stable/introduction/chemistry.html |
| **OpenFermion** | Google's toolkit for fermionic simulation | https://github.com/quantumlib/OpenFermion |
| **Qiskit Nature** | IBM's quantum chemistry package | https://github.com/qiskit-community/qiskit-nature |
| **InQuanto** | Quantinuum's chemistry package (integrated into Azure Quantum Elements) | https://www.quantinuum.com/computationalchemistry/inquanto |
| **PySCF** | Classical quantum chemistry (backend for PennyLane) | https://github.com/pyscf/pyscf |
| **Azure Quantum Elements** | Microsoft's chemistry platform | https://quantum.microsoft.com/en-us/quantum-elements/product-overview |

### 4.6 Key References

| Resource | URL |
|----------|-----|
| **Cao et al. "Quantum Chemistry in the Age of Quantum Computing" (2019)** | https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803 |
| **PennyLane Building Molecular Hamiltonians Tutorial** | https://pennylane.ai/qml/demos/tutorial_quantum_chemistry/ |
| **MS+Quantinuum Chemistry Demo Blog** | https://azure.microsoft.com/en-us/blog/quantum/2024/09/10/microsoft-and-quantinuum-create-12-logical-qubits-and-demonstrate-a-hybrid-end-to-end-chemistry-simulation/ |
| **MS Chemistry Simulation Paper** | http://aka.ms/ArXivMSFTChemSimPaper |
| **Azure Quantum Resource Estimator: Chemistry Tutorial** | https://learn.microsoft.com/en-us/azure/quantum/tutorial-resource-estimator-chemistry |

---

<a name="task-5"></a>
## Task 5: Circuit-Based Quantum Computing Fundamentals

### 5.1 From Annealing to Gates: Key Differences

| Feature | Quantum Annealing (D-Wave) | Gate-Based QC |
|---------|---------------------------|---------------|
| Model | Adiabatic evolution | Quantum circuit model |
| Operations | Continuous time evolution | Discrete gates |
| Problem type | Optimization (QUBO/Ising) | Universal computation |
| Error correction | Limited | Full QEC possible |
| Universality | Not universal | Universal |
| Your experience | ✅ Strong | 🔶 Learning |

### 5.2 Essential Gates

**Single-qubit gates:**
- **Pauli gates:** $X$ (NOT), $Y$, $Z$ (phase flip)
- **Hadamard:** $H = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1 \\ 1 & -1\end{pmatrix}$ — creates superposition
- **Phase gates:** $S = \begin{pmatrix}1 & 0 \\ 0 & i\end{pmatrix}$, $T = \begin{pmatrix}1 & 0 \\ 0 & e^{i\pi/4}\end{pmatrix}$
- **Rotation gates:** $R_x(\theta)$, $R_y(\theta)$, $R_z(\theta)$

**Two-qubit gates:**
- **CNOT:** Controlled-NOT, fundamental entangling gate
- **CZ:** Controlled-Z
- **SWAP:** Exchanges two qubits
- **$\sqrt{iSWAP}$:** Native gate on some superconducting hardware

**Universal gate sets:**
- {H, T, CNOT} — universal for quantum computation
- {Clifford gates + T} — the standard fault-tolerant gate set
- Clifford gates alone are efficiently simulable classically (Gottesman-Knill theorem)

### 5.3 Key Algorithms to Know

1. **Quantum Phase Estimation (QPE):**
   - Estimates eigenvalues of unitary operators
   - Foundation for quantum chemistry (exact ground-state energy)
   - Requires controlled-U operations and QFT
   - Circuit depth ~O(1/ε) for precision ε

2. **Quantum Fourier Transform (QFT):**
   - Quantum analog of discrete Fourier transform
   - Uses O(n²) gates for n qubits
   - Sub-component of QPE, Shor's algorithm

3. **Grover's Algorithm:**
   - Quadratic speedup for unstructured search
   - O(√N) queries vs. O(N) classical

4. **VQE:** (See Task 2)

5. **Hamiltonian Simulation:**
   - Simulate time evolution $e^{-iHt}$
   - Trotter-Suzuki decomposition, qubitization, quantum signal processing
   - Critical for quantum chemistry beyond VQE

### 5.4 Measurement and Born Rule

- Measurement in computational basis: probability of outcome $|x\rangle$ is $|\langle x|\psi\rangle|^2$
- **Mid-circuit measurements:** Measure some qubits during computation (used in QEC)
- **Measurement-based quantum computing (MBQC):** Computation via sequential measurements on entangled cluster states — this is Microsoft's approach with topological qubits!

### 5.5 Key Resources

| Resource | URL | Type |
|----------|-----|------|
| **PennyLane Codebook** | https://pennylane.ai/codebook | Full curriculum with exercises |
| **Quantum Katas by Microsoft** | https://quantum.microsoft.com/en-us/tools/quantum-katas | 26 modules, Q# and interactive |
| **Qiskit Textbook** | https://learning.quantum.ibm.com/catalog/courses | IBM's online textbook |
| **PennyLane Codebook: Intro to QC** | https://pennylane.ai/codebook/introduction-to-quantum-computing | 3 topics |
| **PennyLane Codebook: QPE** | https://pennylane.ai/codebook/quantum-phase-estimation | 4 topics |

**YouTube Recommendations:**
- "Quantum Computing Course" — MIT OpenCourseWare 8.370 (full semester)
- "Introduction to Quantum Computing" — IBM Qiskit YouTube playlist
- "Quantum Computing for Computer Scientists" — Microsoft Research (Andrew Helwer talk, ~1 hour, highly recommended)

---

<a name="task-6"></a>
## Task 6: Microsoft Quantum Stack Deep Dive

### 6.1 Microsoft's Quantum Roadmap

**Three Implementation Levels:**

| Level | Name | Description | Status |
|-------|------|-------------|--------|
| **Level 1** | Foundational | Noisy physical qubits | Current state of industry |
| **Level 2** | Resilient | Reliable logical qubits | **Active work (MS + partners)** |
| **Level 3** | Scale | Quantum supercomputers | Future goal |

**Six Milestones:**
1. ✅ **Create & Control Majoranas** — Accomplished May 2023
2. ✅ **Hardware-Protected (Topological) Qubit** — Demonstrated Feb 2025 (Majorana 1)
3. 🔶 **Quantum error detection and correction demonstrations** — In progress
4. 🔶 **Fault-tolerant prototype** — DARPA-funded, "years not decades"
5. ⬜ **Scalable fault-tolerant quantum computer**
6. ⬜ **Quantum supercomputer** (1M+ reliable qubits, >1 rQOPS/sec)

### 6.2 Majorana 1 — The Topological Qubit (Feb 2025)

**What it is:** World's first QPU powered by a Topological Core. Built with a breakthrough class of materials called **topoconductors**.

**Key physics:**
- **Topoconductors:** Indium arsenide (semiconductor) + aluminum (superconductor), cooled to near absolute zero with magnetic fields
- **Majorana Zero Modes (MZMs):** Quasiparticles at the ends of topological superconducting nanowires
- **Parity-based encoding:** Quantum information stored as even/odd electron parity; protected by topology
- **Measurement:** Quantum dot coupled to nanowire ends; microwave reflectometry reads parity state
- **Error rate:** Initial measurement error ~1%, with clear paths to improvement
- **Stability:** Parity flip occurs only once per millisecond on average

**The Tetron architecture:**
- Single qubit = two parallel topological nanowires + perpendicular trivial superconductor
- 4 MZMs per qubit (hence "tetron")
- All operations via measurements (digital control, not analog rotation)
- Roadmap: 1 qubit → 2 qubits (braiding) → 4×2 array (error detection) → 27×13 array (QEC)

**Why it matters for your interview:**
- Microsoft's approach is **measurement-based**, not gate-based in the traditional sense
- Custom **Floquet codes** reduce QEC overhead ~10x vs. standard surface codes
- Designed to scale to **1 million qubits on a single chip**
- DARPA selected Microsoft for final phase of US2QC benchmarking program

### 6.3 Azure Quantum Platform

**Components:**
1. **Q# Programming Language:** High-level, hardware-agnostic quantum language
2. **Quantum Development Kit (QDK):** Open-source SDK for VS Code
3. **Azure Quantum Workspace:** Cloud service to submit jobs to real QPUs
4. **Hardware Partners:** Quantinuum, IonQ, PASQAL, Rigetti
5. **Resource Estimator:** Estimates physical resources for fault-tolerant algorithms
6. **Azure Quantum Elements:** Chemistry & materials science platform (HPC + AI + quantum)

### 6.4 Microsoft Quantum Resource Estimator

**What it does:** Estimates physical resources (qubits, runtime, T-factory count) needed to run quantum algorithms on future fault-tolerant hardware.

**Key features:**
- Customizable qubit parameters (error rates, gate times)
- Multiple QEC schemes (surface code, custom codes)
- Space-time tradeoff visualization
- Supports Q# and Qiskit input
- **Free, no Azure account required**

**Why it matters:** The MS+Quantinuum chemistry simulation paper and the MS interview will likely involve resource estimation discussions. Key tutorial: https://learn.microsoft.com/en-us/azure/quantum/tutorial-resource-estimator-chemistry

### 6.5 Key People at Microsoft Quantum

| Person | Role | Focus |
|--------|------|-------|
| **Krysta Svore** | Technical Fellow, VP Advanced Quantum Development | Quantum algorithms, logical qubits, chemistry |
| **Chetan Nayak** | Technical Fellow, CVP Quantum Hardware | Topological qubits, Majorana 1 |
| **Matthias Troyer** | Technical Fellow | Quantum algorithms, computational physics |
| **Michael Freedman** | Founder of Station Q | Topological quantum computing theory |

### 6.6 Microsoft's Chemistry/Physics Focus (directly relevant to CS-632)

The job description for CS-632 mentions:
- Exploring chemistry & physics applications for **early fault-tolerant quantum computers**
- Developing/evaluating algorithms (VQE, QPE variants for NISQ-to-FTQC transition)
- Investigating quantum primitives (logical qubits, surface-code operations)
- Applications: **molecular simulation, materials discovery, strongly correlated systems**

**Key MS publications to read:**
| Paper | URL |
|-------|-----|
| MS+Quantinuum 12 logical qubits paper | http://aka.ms/ArXivMSFTLQPaper |
| MS chemistry simulation paper | http://aka.ms/ArXivMSFTChemSimPaper |
| Majorana 1 Nature paper | https://aka.ms/MSQuantumNaturePaper |
| Topological architecture arXiv paper | https://aka.ms/MSBrandArXivTopo |

### 6.7 Resources

| Resource | URL |
|----------|-----|
| **Azure Quantum Overview** | https://learn.microsoft.com/en-us/azure/quantum/overview-azure-quantum |
| **Resource Estimator** | https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation |
| **Majorana 1 Blog Post** | https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/ |
| **MS+Quantinuum Chemistry Blog** | https://azure.microsoft.com/en-us/blog/quantum/2024/09/10/microsoft-and-quantinuum-create-12-logical-qubits-and-demonstrate-a-hybrid-end-to-end-chemistry-simulation/ |
| **MS Quantum Blog (all posts)** | https://azure.microsoft.com/en-us/blog/quantum/ |
| **Q# Getting Started** | https://learn.microsoft.com/en-us/azure/quantum/qsharp-quickstart |
| **Quantum Katas** | https://quantum.microsoft.com/en-us/tools/quantum-katas |
| **MS Research Podcast (Chetan Nayak on topological qubits)** | https://aka.ms/MSRPodTopo |
| **Microsoft Research Quantum Computing** | https://www.microsoft.com/research/research-area/quantum-computing/ |

---

<a name="task-7"></a>
## Task 7: 7-Day Study Plan

### Overview: Prioritized by relevance to CS-632

**Your strengths:** D-Wave annealing, PennyLane basics, transmon simulation, optimization  
**Gaps to fill:** VQE, QEC/surface codes, computational chemistry, circuit QC, Microsoft stack

---

### Day 1: VQE Foundations (HIGH PRIORITY)
**Morning (3-4 hours):**
- Read Sections I-III of the VQE review paper (https://arxiv.org/abs/2111.05176) — focus on overview, Hamiltonian representation, and ansätze
- Watch: "VQE Explained" on PennyLane YouTube channel (~20 min)

**Afternoon (3-4 hours):**
- Complete PennyLane VQE tutorial hands-on: https://pennylane.ai/qml/demos/tutorial_vqe/
- Complete PennyLane molecular Hamiltonians tutorial: https://pennylane.ai/qml/demos/tutorial_quantum_chemistry/
- **Deliverable:** Run VQE for H₂ and LiH in PennyLane; understand Jordan-Wigner mapping

**Evening (1-2 hours):**
- Read PennyLane chemistry docs: https://docs.pennylane.ai/en/stable/introduction/chemistry.html
- Write notes on: UCCSD ansatz, active space selection, measurement strategies

---

### Day 2: Circuit-Based QC Fundamentals
**Morning (3-4 hours):**
- PennyLane Codebook: "Introduction to Quantum Computing" module (https://pennylane.ai/codebook/introduction-to-quantum-computing)
- PennyLane Codebook: "Single-Qubit Gates" module (first 4 topics)
- Focus on: Bloch sphere, rotation gates, Hadamard, measurement

**Afternoon (3-4 hours):**
- PennyLane Codebook: "Circuits with Many Qubits" module
- PennyLane Codebook: "Quantum Phase Estimation" module (all 4 topics)
- **Deliverable:** Understand QPE algorithm and how it differs from VQE

**Evening (1-2 hours):**
- Watch: "Quantum Computing for Computer Scientists" (Microsoft Research, Andrew Helwer, YouTube)
- Review: How your annealing intuition maps to gate-based circuits

---

### Day 3: Quantum Error Correction & Surface Codes (HIGH PRIORITY)
**Morning (3-4 hours):**
- Read: Fowler et al. Sections 1-4 of "Surface codes" paper (https://arxiv.org/abs/1208.0928) — stabilizers, surface code layout, logical qubits
- Quantum Katas: "QEC: Bit Flip, Phase Flip, and Shor Codes" module (https://quantum.microsoft.com/en-us/tools/quantum-katas)

**Afternoon (3-4 hours):**
- PennyLane Codebook: "Quantum Error Correction" module (https://pennylane.ai/codebook/quantum-error-correction)
- Study: Lattice surgery, magic state distillation (Fowler et al. Sections 5-7)
- **Deliverable:** Be able to explain surface code error correction on a whiteboard

**Evening (1-2 hours):**
- Read: Microsoft's Floquet codes blog post — understand why MS's approach reduces overhead ~10x
- Read: Microsoft 4D codes blog (https://azure.microsoft.com/en-us/blog/quantum/2025/06/19/microsoft-advances-quantum-error-correction-with-a-family-of-novel-four-dimensional-codes/)

---

### Day 4: Microsoft Quantum Stack & Topological Qubits (HIGH PRIORITY)
**Morning (3-4 hours):**
- Read thoroughly: Majorana 1 blog post (https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/)
- Listen: MS Research Podcast with Chetan Nayak (https://aka.ms/MSRPodTopo)
- Study: Topological qubit physics — MZMs, parity encoding, tetron architecture

**Afternoon (3-4 hours):**
- Set up QDK in VS Code; run a simple Q# program
- Explore Azure Quantum Resource Estimator: https://learn.microsoft.com/en-us/azure/quantum/quickstart-microsoft-resources-estimator
- Run the chemistry resource estimation tutorial: https://learn.microsoft.com/en-us/azure/quantum/tutorial-resource-estimator-chemistry
- **Deliverable:** Understand MS's roadmap, be able to discuss Majorana 1 in depth

**Evening (1-2 hours):**
- Read: MS+Quantinuum 12 logical qubits and chemistry simulation blogs
- Prepare talking points on how HPC + quantum + AI work together (Azure Quantum Elements)

---

### Day 5: Computational Chemistry Deep Dive
**Morning (3-4 hours):**
- Read: Cao et al. "Quantum Chemistry in the Age of Quantum Computing" — Sections on near-term algorithms and electronic structure (https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803)
- Study: Bravyi-Kitaev transformation, qubit tapering, active space selection in detail

**Afternoon (3-4 hours):**
- Hands-on: Implement VQE for H₂O using PennyLane qchem (active space reduction)
- Explore: ADAPT-VQE concept and implementation
- Hands-on: Use PennyLane's OpenFermion backend: `qchem.molecular_hamiltonian(molecule, method='pyscf')`
- **Deliverable:** End-to-end chemistry simulation for a 3+ atom molecule

**Evening (1-2 hours):**
- Review MS chemistry simulation paper (http://aka.ms/ArXivMSFTChemSimPaper) — understand the hybrid workflow
- Prepare to discuss: "How would you simulate molecule X on a quantum computer?"

---

### Day 6: Advanced Topics & Practice Problems
**Morning (3-4 hours):**
- Study: Hamiltonian simulation (Trotter-Suzuki decomposition, product formulas)
- PennyLane Codebook: "Hamiltonian Simulation" module (https://pennylane.ai/codebook/hamiltonian-simulation)
- Study: Classical shadows protocol (used in MS chemistry demo)

**Afternoon (3-4 hours):**
- Practice interview questions (Section 1.3 above) — verbal and whiteboard
- Focus on: "Explain VQE from scratch," "Draw a surface code," "What is a topological qubit?"
- Prepare your research presentation: D-Wave work → bridge to chemistry applications
- **Deliverable:** 5-minute pitch of your background, connecting to MS's mission

**Evening (1-2 hours):**
- Review: Error mitigation techniques (zero-noise extrapolation, probabilistic error cancellation)
- Read: Recent MS quantum blog posts for latest developments

---

### Day 7: Final Review & Mock Interview
**Morning (3-4 hours):**
- Review all notes from Days 1-6
- Re-read: Majorana 1 blog, MS+Quantinuum chemistry blog, MS roadmap
- Practice explaining: VQE, surface codes, Jordan-Wigner transformation, QPE — in 2 minutes each

**Afternoon (3-4 hours):**
- Mock interview with a friend/colleague covering:
  - "Tell me about your research" (3-5 min)
  - "How would you design a VQE experiment for a new molecule?" (whiteboard)
  - "Explain quantum error correction to me" (5 min)
  - "What excites you about Microsoft's approach?" (2 min)
  - "What would you want to work on during your internship?" (2 min)

**Evening (1-2 hours):**
- Prepare questions to ASK the interviewer:
  - "What molecules/materials are you most excited about simulating?"
  - "How does the team balance near-term NISQ algorithms vs. fault-tolerant approaches?"
  - "How do you see topological qubits changing the resource estimation landscape for chemistry?"
  - "What role does classical shadows play in your current chemistry workflow?"
  - "How does the Azure Quantum Elements pipeline integrate quantum and HPC?"
- Final: Rest and prepare mentally. You have strong fundamentals — the D-Wave + transmon experience is a genuine asset.

---

## Quick Reference: All URLs Collected

### Microsoft Resources
| Resource | URL |
|----------|-----|
| MS Research Quantum Computing | https://www.microsoft.com/research/research-area/quantum-computing/ |
| Microsoft Quantum Roadmap | https://quantum.microsoft.com/en-us/our-story |
| Azure Quantum Overview | https://learn.microsoft.com/en-us/azure/quantum/overview-azure-quantum |
| Azure Quantum Resource Estimator | https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation |
| Quantum Katas | https://quantum.microsoft.com/en-us/tools/quantum-katas |
| MS Quantum Blog | https://azure.microsoft.com/en-us/blog/quantum/ |
| Majorana 1 Blog | https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/ |
| MS+Quantinuum Chemistry Blog | https://azure.microsoft.com/en-us/blog/quantum/2024/09/10/microsoft-and-quantinuum-create-12-logical-qubits-and-demonstrate-a-hybrid-end-to-end-chemistry-simulation/ |
| MS 4D QEC Codes Blog | https://azure.microsoft.com/en-us/blog/quantum/2025/06/19/microsoft-advances-quantum-error-correction-with-a-family-of-novel-four-dimensional-codes/ |
| Q# Quick Start | https://learn.microsoft.com/en-us/azure/quantum/qsharp-quickstart |
| Chemistry Resource Estimation Tutorial | https://learn.microsoft.com/en-us/azure/quantum/tutorial-resource-estimator-chemistry |
| QDK Overview | https://learn.microsoft.com/en-us/azure/quantum/qdk-main-overview |
| MS Research Podcast (Chetan Nayak) | https://aka.ms/MSRPodTopo |
| Majorana 1 Nature Paper | https://aka.ms/MSQuantumNaturePaper |
| Topological Architecture ArXiv | https://aka.ms/MSBrandArXivTopo |
| MS Logical Qubits Paper | http://aka.ms/ArXivMSFTLQPaper |
| MS Chemistry Simulation Paper | http://aka.ms/ArXivMSFTChemSimPaper |
| Azure Quantum Elements | https://quantum.microsoft.com/en-us/quantum-elements/product-overview |

### Learning Platforms
| Resource | URL |
|----------|-----|
| PennyLane VQE Tutorial | https://pennylane.ai/qml/demos/tutorial_vqe/ |
| PennyLane Molecular Hamiltonians | https://pennylane.ai/qml/demos/tutorial_quantum_chemistry/ |
| PennyLane Chemistry Docs | https://docs.pennylane.ai/en/stable/introduction/chemistry.html |
| PennyLane Codebook (full) | https://pennylane.ai/codebook |
| PennyLane Codebook: QEC | https://pennylane.ai/codebook/quantum-error-correction |
| PennyLane Codebook: QPE | https://pennylane.ai/codebook/quantum-phase-estimation |
| PennyLane Codebook: VQA | https://pennylane.ai/codebook/variational-quantum-algorithms |
| PennyLane Codebook: Hamiltonian Simulation | https://pennylane.ai/codebook/hamiltonian-simulation |

### Key Papers
| Paper | URL |
|-------|-----|
| Peruzzo et al. "VQE on a quantum processor" (2014) | https://arxiv.org/abs/1304.3061 |
| Tilly et al. "VQE: review of methods and best practices" (2022) | https://arxiv.org/abs/2111.05176 |
| Fowler et al. "Surface codes: Towards practical large-scale QC" (2012) | https://arxiv.org/abs/1208.0928 |
| Cao et al. "Quantum Chemistry in the Age of QC" (2019) | https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803 |

### Reference Sites
| Resource | URL |
|----------|-----|
| Error Correction Zoo: Surface Code | https://errorcorrectionzoo.org/c/surface |
| OpenFermion (Google) | https://github.com/quantumlib/OpenFermion |
| Qiskit Nature | https://github.com/qiskit-community/qiskit-nature |
| PySCF | https://github.com/pyscf/pyscf |

---

## Your Competitive Advantages

1. **D-Wave experience → Optimization mindset:** VQE is fundamentally an optimization problem. Your MILP/QUBO expertise translates directly.
2. **Transmon simulation → Hardware understanding:** You know qubit physics (T1/T2, decoherence, coupling) from first principles.
3. **PennyLane proficiency:** Already familiar with the framework used in many chemistry tutorials.
4. **Benchmarking experience:** Your quantum vs. classical solver comparisons are directly relevant to MS's focus on demonstrating quantum advantage.
5. **EPFL + CERN pedigree:** Strong academic credentials in a top quantum program.

**Key narrative:** "I've worked with quantum hardware at both the physical level (transmon simulations) and the application level (D-Wave optimization). I'm now ready to bridge these into the most impactful application of quantum computing: chemistry and materials simulation. Microsoft's approach — combining logical qubits, HPC, and AI — aligns perfectly with my background in hybrid quantum-classical methods."

---

*Good luck with your interview, Edoardo! 🎯*
