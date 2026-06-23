"""
LMR (Lloyd-Mohseni-Rebentrost) sample-based implementation of 
    e^(-i alpha |Phi><Phi|).

    S. Lloyd, M. Mohseni, P. Rebentrost,
    "Quantum principal component analysis",
    Nat. Phys. 10, 631-633 (2014).

Given lmr_num_copies copies of |Phi> (on separate n-qubit registers) and the working
n-qubit register R in some state rho_R, this module applies

         exp(-i alpha |Phi><Phi|)  
                    = I - (1 - e^{-i alpha}) |Phi><Phi|  
                    (since |Phi><Phi|^2 = |Phi><Phi|)

to rho_R.

Idea: 

    tr_C [ exp(-i S_{RC} dt) (rho_R otimes |Phi><Phi|_C) exp(+i S_{RC} dt) ]
        = rho_R - i dt [|Phi><Phi|, rho_R] + O(dt^2)

Repeating kappa times with dt = alpha / lmr_num_copies accumulates total time alpha; 
the residual error is O(alpha^2 / lmr_num_copies) in trace distance.

Output of this module is a QuantumCircuit; tracing out the copies and the
ancilla is the *caller's* responsibility (the circuit itself is unitary).

Qubit layout in the returned circuit (n + n * lmr_num_copies + 1 qubits total):
    working register R indices: (0, n)                
    j-th copy indices:          (n + j*n, n + (j+1)*n)        (j = 0..lmr_num_copies-1)
    1-qubit ancilla index:      n*(1+lmr_num_copies)                    
"""

from __future__ import annotations

from qiskit import QuantumCircuit



# ---------------------------------------------------------------------------
# Computing the trace distance of two density matrices: D_{tr}(rho_A,rho_B)
# ---------------------------------------------------------------------------
def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        ev = np.linalg.eigvalsh(rho - sigma)
        return 0.5 * float(np.sum(np.abs(ev)))


# ---------------------------------------------------------------------------
# Single partial register SWAP:  exp(-i S_{R,C} dt)
# ---------------------------------------------------------------------------
def _swap_hamiltonian(
    qc: QuantumCircuit,
    R_qubits: list[int],
    C_qubits: list[int],
    ancilla: int,
    dt: float,
) -> None:
    """
    Append exp(-i S_{R,C} dt) to qc, where S_{R,C} is the register-SWAP that
    maps |x>_R |y>_C to |y>_R |x>_C.

    Implementation.  S_{R,C}^2 = I, so S has eigenvalues +-1 correspond to 
    two eigenspaces V^+ and V^-. Let P^+ and P^- be the projection to these
    eigenspaces. We have that P^+ = (I+S)/2 and P^- = (I-S)/2. Then we have 
    S = P^+ + (-1) P^-. Therefore, we have e^{-iS dt} = e^{-i dt} P^+ + e^{i dt} P^
    We bin the joint state |psi>_RC into the two eigenspaces by entangling them with a Hadamard
    ancilla through a controlled register-SWAP; apply R_x(2 dt) on the ancilla
    so the V^+ eigenspace picks up e^{-i dt} and the V^- eigenspace picks up
    e^{+i dt}; then uncompute the entanglement, leaving the ancilla in |0>.

    Cost: 2 Hadamards + 2n CSWAPs + 1 R_x per.
    """
    if len(R_qubits) != len(C_qubits):
        raise ValueError("R and C must hold the same number of qubits.")

    qc.h(ancilla)
    for r, c in zip(R_qubits, C_qubits):
        qc.cswap(ancilla, r, c)
    qc.rx(2.0 * dt, ancilla)                 # +1 eigenspace -> e^{-i dt}, -1 -> e^{+i dt}
    for r, c in zip(R_qubits, C_qubits):
        qc.cswap(ancilla, r, c)
    qc.h(ancilla)


# ---------------------------------------------------------------------------
# Coherent LMR simulation
# ---------------------------------------------------------------------------
def LMR_simulation(
        state_num_qubits: int,
        lmr_num_copies: int, 
        alpha: float, 
    ) -> QuantumCircuit:
    """
    Build the LMR approximation of exp(-i alpha |Phi><Phi|) using
    lmr_num_copies copies of |Phi> as a resource.

    Input
    ----------
    alpha : float
        The desired phase angle.
    lmr_num_copies : int >= 1
        Number of LMR steps; one copy of |Phi> is consumed per step.
        Trace-distance error scales as O(alpha^2 / lmr_num_copies).
    state_num_qubits : int >= 1
        Qubits per register (so |Phi> is an (state_num_qubits)-qubit state).

    Output
    -------
    QuantumCircuit on state_num_qubits*(1+lmr_num_copies) + 1 qubits.
    Pre-conditions on the caller:
        - qubits [0, state_num_qubits)              hold rho_R, the working state;
        - qubits [state_num_qubits, state_num_qubits*(1+lmr_num_copies))  
                            hold lmr_num_copies independent copies of |Phi>, 
                            one per state_num_qubits-qubit block;
        - qubit  state_num_qubits * (1 + lmr_num_copies)                  is in |0>.
    Post-conditions after running:
        - the ancilla is restored to |0>;
        - the copies are in some state entangled with R; the caller must trace
          them out (e.g. via partial_trace, or by measurement-and-discard).
    """
    if lmr_num_copies < 1:
        raise ValueError("number of copies must be a positive integer.")
    if state_num_qubits < 1:
        raise ValueError("number of qubits of the main state must be a positive integer.")

    total = state_num_qubits * (1 + lmr_num_copies) + 1
    qc = QuantumCircuit(total, name=f"LMR(alpha={alpha:.4f}, kappa={lmr_num_copies})")

    state_register = list(range(state_num_qubits))
    ancilla = state_num_qubits * (1 + lmr_num_copies)
    dt = alpha / lmr_num_copies

    for k in range(lmr_num_copies):
        C = list(range(state_num_qubits + k * state_num_qubits, state_num_qubits + (k + 1) * state_num_qubits))
        _swap_hamiltonian(qc, state_register, C, ancilla, dt)

    return qc


# ---------------------------------------------------------------------------
# Self-test: trace-distance convergence  d_tr(rho_LMR, rho_exact) ~ 1/kappa.
#
# First test, simulate LMR hamiltonian coherently and then trace out extra 
# registers and copies.Note that this results in exponential-size 
# matrix manipulation.
#
# Second test, simulates the LMR channel step-by-step (tracing out the copy after every
# step), which is what the algorithm virtually does, and avoids exponential
# blow-up in the simulation cost as kappa grows.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    import math
    import numpy as np
    from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
    from qiskit.circuit.library import StatePreparation

    # Number of qubits
    n = 1  

    # Setting up states phi, psi and an angle alpha for testing 
    # e^{-i alpha |phi><phi|} |psi><psi| e^{i alpha |phi><phi|}
    phi = np.array([math.cos(np.pi/8),math.sin(np.pi/8)], complex)
    ketbra_phi = np.outer(phi,phi.conj())
    alpha = math.pi/23
    psi = np.array([math.cos(np.pi/3.2),math.sin(np.pi/3.2)], complex)
    ketbra_psi = np.outer(psi,psi.conj())


    # Computing U = e^(-i alpha ketbra_phi) = I - (1 - e^{-i alpha}) |Phi><Phi|
    U = np.eye(2) 
    U = U - (1 - np.exp(- 1j * alpha)) * ketbra_phi

    # Computing U |psi><psi| U*
    exact_result = U @ (ketbra_psi @ U.conj())


    # ------------------------------------------------------------------
    #  FIRST TEST
    # ------------------------------------------------------------------
    print(f"{'kappa':>6} | {'trace dist':>12} | {'alpha^2/kappa':>10}")
    print("-" * 38)
    for kappa in [1, 2, 4, 8, 12, 16]:
        total = n * (1 + kappa) + 1
 
        # ---- build the full circuit, with all input states prepared ----
        full = QuantumCircuit(total, name=f"LMR_test(kappa={kappa})")
 
        # Working register R (qubits [0, n)) gets |psi>.
        full.append(StatePreparation(psi), list(range(n)))
 
        # Each of the kappa copy registers gets a fresh |Phi>.
        for k in range(kappa):
            start = n + k * n
            full.append(StatePreparation(phi), list(range(start, start + n)))
 
        # The ancilla (qubit n*(1+kappa)) starts in |0> by default; nothing to do.
 
        # Apply LMR_simuation on top of the prepared state.
        lmr_qc = LMR_simulation(n, kappa, alpha)
        full.compose(lmr_qc, inplace=True)
 
        # ---- simulate ----
        sv = Statevector.from_label("0" * total).evolve(full)
 
        # ---- discard the copy registers (qubits [n, n*(1+kappa))) and the ancilla;
        #      keep only the working register R (qubits [0, n)).
        discard = list(range(n, total))
        simulated_result = partial_trace(sv, discard)
 
        # ---- trace distance from the exact target ----
        td = trace_distance(simulated_result.data, exact_result)
        print(f"{kappa:>5d} | {td:>12.6f} | {alpha**2/kappa:>9.6f}")
    

    # Separate the two tests outputs
    print("#" * 50)

    # ------------------------------------------------------------------
    #  SECOND TEST
    # ------------------------------------------------------------------

    # ---- pre-built density matrices for a fresh copy and a |0> ancilla ----
    phi_rho = DensityMatrix(np.outer(phi, phi.conj()))
    ancilla = DensityMatrix(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex))

    print(f"{'kappa':>6} | {'trace dist':>12} | {'alpha^2/kappa':>10}")
    print("-" * 38)
    for kappa in [1, 2, 4, 8, 16, 32, 64, 128]:
        dt = alpha / kappa

        # Single-step circuit acting on (R, one fresh copy, ancilla) -- 2n+1 qubits.
        one_step_circ = QuantumCircuit(2 * n + 1)
        _swap_hamiltonian(
            one_step_circ,
            list(range(n)),            # R
            list(range(n, 2 * n)),     # C: one fresh copy
            2 * n,                     # ancilla
            dt,
        )

        # Initial working-register state.
        rho_R = DensityMatrix(np.outer(psi, psi.conj()))

        # kappa steps: bring in a fresh copy and a fresh ancilla, apply the
        # partial SWAP, then trace them both out before the next step.
        for _ in range(kappa):
            joint = ancilla.tensor(phi_rho).tensor(rho_R)
            joint = joint.evolve(one_step_circ)
            rho_R = partial_trace(joint, list(range(n, 2 * n + 1)))

        td = trace_distance(rho_R.data, exact_result)
        print(f"{kappa:>6d} | {td:>12.6f} | {alpha**2/kappa:>10.6f}")
    



