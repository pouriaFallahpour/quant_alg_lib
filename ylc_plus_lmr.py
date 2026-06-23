"""
Implementing YLC amplification by simulating S_s using LMR_simulation 
and many copies of the initial state.

"""

from __future__ import annotations
import math
import numpy as np

from typing import Callable

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

from qiskit.qasm2 import dumps    # OpenQASM 2 (most readable, like the old .qasm())

from ylc_amplifier import _ylc_phase_schedule, _apply_St


# ---------------------------------------------------------------------------
# Modular YLC amplifier to handle LMR simulation of the reflection S_s
# ---------------------------------------------------------------------------
def YLC_plus_LMR(
    state_num_qubits: int,
    lmr_num_copies: int,
    generic_lmr: Callable[[int, int, float], QuantumCircuit],
    O_Good: QuantumCircuit,
    lam: float,
    delta: float = 1e-2,
) -> tuple[QuantumCircuit, dict]:
    """
    Input
    ----------
    num_qubits : an integer
        the number of qubits of the main state.
    lmr_num_copies : an integer
        The number of copies of the state.
    generic_lmr : a callable function
        An alias for LMR_simulation
    O_Good : QuantumCircuit on n+1 qubits
        Convention: qubits 0..n-1 are the state register R, qubit n is the
        1-qubit Good-indicator ancilla a.  Acts as
            |Good>_R |b>_a -> |Good>_R |b XOR 1>_a
            |Bad>_R  |b>_a -> |Bad>_R  |b>_a
    lam: a rational
        The lower bound for |amplitude of |Good>|^2
    delta: a rational
        The error in the final amplified state

    Output
    -------
    qc : a QuantumCircuit object.
        Set alias n for  num_qubits and num_copies for lmr_num_copies for simplicity.
        The {0,...,n-1} register contains the main state.
        The {n} register is ancilla for YLC 
        First iteration of YLC: the {n+1,...,2n} & {2n+1,...,3n} & .... & {num_copies * n + 1, (num_copies + 1) * n} registers 
        Second iteration of YLC: the {(num_copies + 1) * n + 1,...,(num_copies + 2) * n} 
                                    & .... & {(2 * num_copies) * n + 1,..., (2 * num_copies + 1) * n} registers 
                        .
                        .
                        .

        j-th iteration of YLC: the {( (j-1) * num_copies + 1) * n + 1,...,( (j-1) * num_copies + 2) * n} 
                                    & .... & {( j * num_copies) * n + 1,..., ( j * num_copies + 1) * n} registers 
                        .
                        .
                        .
        l-th iteration of YLC: the {( (l-1) * num_copies + 1) * n + 1,...,( (l-1) * num_copies + 2) * n} 
                                    & .... & {( l * num_copies) * n + 1,..., ( l * num_copies + 1) * n} registers

        Ancilla for j-th iteration: ( l * num_copies + 1) * n + j

        Total number of qubits: ( l * num_copies + 1) * n + l
    """
 
    # Probe the provider once to verify the number of qubits match.
    sample_circ = generic_lmr(state_num_qubits, lmr_num_copies, 0.0)
    lmr_num_qubits = sample_circ.num_qubits
    if lmr_num_qubits !=  state_num_qubits * (1 + lmr_num_copies) + 1:
        raise ValueError(
            f"S_s_provider returned a circuit with {sample_circ.num_qubits} qubits, "
            f"fewer than n = {state_num_qubits * (1 + lmr_num_copies) + 1}."
        )
 
    # Type check
    if O_Good.num_qubits != state_num_qubits + 1:
        raise ValueError(f"O_Good must act on n+1 = {state_num_qubits+1} qubits.")
    if not (0.0 < lam <= 1.0):
        raise ValueError("lam must lie in (0, 1].")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")
 
    # --- iteration count and phase schedule ---
    L_real = math.log(2.0 / delta) / math.sqrt(lam)
    L = math.floor(L_real) + 1
    if L % 2 == 0:
        L += 1
    l = (L - 1) // 2
    alphas, betas = _ylc_phase_schedule(L, delta)
 

    # Computing the right amount of qubits. Consult function output type
    O_Good_gate = O_Good.to_gate(label="O_G")
    state_register = list(range(state_num_qubits))
    reflection_ancilla= [state_num_qubits]    # ancilla for _apply_St
    qc = QuantumCircuit(  ( l * lmr_num_copies + 1) * state_num_qubits + l + 1, name=f"YLC with LMR")
 
    for j in range(l):
        # ============================================================
        # S_t(beta_j) 
        # ============================================================
        _apply_St(qc, betas[j], state_register, reflection_ancilla, O_Good_gate)
 
        # ============================================================
        # S_s(alpha_j) via LMR.
        # ============================================================

        lmr_j = generic_lmr(state_num_qubits, lmr_num_copies, alphas[j])

        # j starts from 0, so to match correctly with the register structure, use j+1
        lmr_j_copies = list(range(( (j) * lmr_num_copies + 1) * state_num_qubits + 1 , ( (j+1) * lmr_num_copies + 1) * state_num_qubits + 1))
        lmr_j_ancilla = [( l * lmr_num_copies + 1) * state_num_qubits + (j + 1)]
        lmr_j_registers = state_register + lmr_j_copies  +  lmr_j_ancilla
        print(f"{j+1}-th LMR iteration acts on registers: {lmr_j_registers}")
        qc.compose(lmr_j, qubits=lmr_j_registers, inplace=True)
 
 
    metadata = { 
        "l": l,
        "alphas": alphas,
        "betas": betas,
    }

    return qc, metadata

# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------------
    #  Test state is sin(theta)|0> + cos(theta)|1>
    # ------------------------------------------------------------------
    theta = math.pi/3                                     
    g_true = math.cos(theta/2)   # We have cos(pi/6) 
    lam = 0.7                  # the lower bound on g_true^2, make sure this is a correct lower bound
    delta = 0.1
    
    num_qubits = 1
    A     = QuantumCircuit(1, name="A");  A.ry(2.0 * theta, 0)
    A_inv = QuantumCircuit(1, name="Adagger"); A_inv.ry(-2.0 * theta, 0)
    O_Good = QuantumCircuit(2, name="O_Good")
    O_Good.cx(0, 1)   
    main_state = Statevector.from_label("0" * num_qubits).evolve(A)
    print(f"\n The main state for test : {main_state.data}\n")

    print("#" * 50)
    print(f"TEST FOR YLC AMPLIFICATION WITH COPIES")
    print(f"Note that delta cannot be exactly fixed in the copy-based method since LMR only offers asymptotic convergence.\n")
    print(f"Note that error of each LMR iteration can propagate to the next rounds and amplify the total error.\n This is because we do not have a concrete LMR estimation.\n Therefore, the test works well for large number of copies, which are difficult to simulate.\n")
    from lmr_simulation import LMR_simulation
    
    num_copies = 9
    circ_with_copies, metadata = YLC_plus_LMR(
        num_qubits,
        num_copies,
        LMR_simulation,
        O_Good,
        lam,
        delta
    )
    l = metadata.get("l",0)


    state_circ = Statevector.from_label("0" * num_qubits).evolve(A)

    print("\n")

    total_copies = l * num_copies
    total_qubits = total_copies * num_qubits
    copy_prep_circ = QuantumCircuit(total_qubits)
    for i in range(total_copies):
        start = i * num_qubits
        copy_prep_circ.append(A, list(range(start, start + num_qubits)))
    many_copies = Statevector.from_label("0" * total_qubits).evolve(copy_prep_circ)

    init_state = Statevector.from_label("0" * num_qubits).evolve(A)
    reflection_ancilla = Statevector.from_label("0")
   

    # The ordering is correct
    init_refl = reflection_ancilla.tensor(init_state)
    init_refl_copies = many_copies.tensor(init_refl)
    lmr_ancilla = Statevector.from_label("0" * l)
    input = lmr_ancilla.tensor(init_refl_copies)
    output = input.evolve(circ_with_copies)

    # compute the labels of extra registers including the ancilla and copies to be discarded
    A = set(range(( l * num_copies + 1) * num_qubits + (l + 1)))
    B = set(range(num_qubits))
    C = sorted(A - B)

    final = partial_trace(output, C)
    print(f"the amplified state is: \n {final.data}\n")
    print(f"total number of qubits: {output.num_qubits}")
    print(f"depth of the circuit: {circ_with_copies.depth()}")
    
    

