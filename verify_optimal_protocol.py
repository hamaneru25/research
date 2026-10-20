"""Verify a deterministic optimum using Gaussian-integer amplitude arithmetic.

Every nonzero outcome is transformed to the standard GHZ state, or |+> for
one survivor. The checks below use integer-valued real/imaginary coefficients,
with all square-root normalization factors carried in an explicit denominator.
"""
import csv
import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np

from optimize_graph_sensing import CandidateFactory, SURVIVORS, edges_of, matrix_key, witness


def conditional_numerator(survivors, bases, outcomes):
    lost = tuple(i for i in range(3) if i not in survivors)
    amp = np.zeros((2**len(survivors), 2**len(lost)), dtype=complex)
    for sensor_bits in product((0, 1), repeat=3):
        row = sum(sensor_bits[i] << (len(survivors)-1-j) for j, i in enumerate(survivors))
        col = sum(sensor_bits[i] << (len(lost)-1-j) for j, i in enumerate(lost))
        for central_bits in product((0, 1), repeat=3):
            bits = sensor_bits + central_bits
            value = (-1)**sum(bits[u]*bits[v] for u, v in edges_of(246))
            for bit, basis, result in zip(central_bits, bases, outcomes):
                if bit:
                    value *= result if basis == "X" else -1j*result
            amp[row, col] += value
    numerator = amp @ amp.conj().T
    trace = int(round(np.trace(numerator).real))
    # Initial graph amplitude 1/8, three Pauli bras contribute 1/sqrt(8).
    probability = Fraction(trace, 512)
    return numerator, trace, probability


def main():
    output = Path("results")
    output.mkdir(exist_ok=True)
    identity = np.eye(2, dtype=complex)
    xgate = np.array([[0, 1], [1, 0]], complex)
    zgate = np.diag([1, -1]).astype(complex)
    sgate = np.diag([1, 1j])
    hnum = np.array([[1, 1], [1, -1]], complex)
    reps = {("Z", 1):(identity, 0, "I"), ("Z", -1):(xgate, 0, "X"),
            ("X", 1):(hnum, 1, "H"), ("X", -1):(hnum@zgate, 1, "H Z"),
            ("Y", 1):(hnum@sgate.conj().T, 1, "H Sdag"),
            ("Y", -1):(hnum@sgate, 1, "H S")}
    factory = CandidateFactory()
    policy = []
    csv_rows = []
    checked = 0
    zero_branches = 0
    for survivors in SURVIVORS:
        k = len(survivors)
        strategy = witness(246, survivors, matrix_key(np.ones((k, k))), factory)
        by_result = {tuple(o["central_result"]):o for o in strategy["outcomes"]}
        total_probability = Fraction(0)
        for result in product((1, -1), repeat=3):
            rho_num, rho_den, probability = conditional_numerator(survivors, strategy["central_bases"], result)
            total_probability += probability
            if not probability:
                zero_branches += 1
                continue
            branch = by_result[result]
            gate_num = np.ones((1, 1), complex)
            h_count = 0
            gate_names = []
            for axis, sign in zip(branch["pulled_back_axes"], branch["axis_signs"]):
                num, count, name = reps[(axis, sign)]
                gate_num = np.kron(gate_num, num)
                h_count += count
                gate_names.append(name)
            transformed = gate_num @ rho_num @ gate_num.conj().T
            denominator = rho_den * 2**h_count
            ghz_twice = np.zeros_like(transformed)
            ghz_twice[0, 0] = ghz_twice[-1, -1] = ghz_twice[0, -1] = ghz_twice[-1, 0] = 1
            # A final known diagonal Clifford on the first survivor aligns the
            # relative GHZ phase; it does not change the pulled-back Z axis/QFI.
            for power in range(4):
                local = np.linalg.matrix_power(sgate, power)
                phase_gate = np.kron(local, np.eye(2**(k-1)))
                aligned = phase_gate @ transformed @ phase_gate.conj().T
                if np.array_equal(2*aligned, denominator*ghz_twice):
                    break
            else:
                raise AssertionError((survivors, result, "Not exactly a corrected GHZ state"))
            branch["phase_gate_power_on_first_survivor"] = power
            branch["probability_exact"] = str(probability)
            branch["gate_products_rightmost_first"] = [
                (f"S^{power} " if i == 0 and power else "") + name
                for i, name in enumerate(gate_names)]
            checked += 1
            csv_rows.append({"survivors": " ".join(f"S{i+1}" for i in survivors),
                             "central_bases_C1_C2_C3": "".join(strategy["central_bases"]),
                             "central_result": " ".join("+" if r == 1 else "-" for r in result),
                             "probability_given_loss": str(probability),
                             "sensor_gate_products": " ; ".join(f"S{i+1}: {g}" for i,g in zip(survivors,branch["gate_products_rightmost_first"]))})
        assert total_probability == 1
        policy.append({"survivors_zero_based": list(survivors), "target": "GHZ+" if k > 1 else "+",
                       "qfim_on_survivors": np.ones((k, k), int).tolist(), **strategy})
    result = {"graph_mask":246, "graph_edges_zero_based":edges_of(246),
              "node_mapping":{"0":"S1","1":"S2","2":"S3","3":"C1","4":"C2","5":"C3"},
              "nonzero_branches_verified_exactly":checked, "zero_probability_branches":zero_branches,
              "gate_convention":"Products act rightmost first; S=diag(1,i). All sensor gates precede phase encoding.",
              "randomized_choices_required":False,
              "qfim_formula":"eta*(1-eta)*I3 + eta**2*ones(3,3)",
              "effective_qfi_formula":"3*eta + 6*eta**2", "generated_photons":6,
              "policies":policy}
    with open(output/"deterministic_optimal_protocol.json","w") as f:
        json.dump(result,f,indent=2)
    with open(output/"deterministic_gate_table.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]))
        writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({"verified_nonzero_outcomes":checked,"verified_zero_outcomes":zero_branches,
                      "all_targets":"exact GHZ+ / +", "randomization_required":False}))


if __name__ == "__main__":
    main()
