"""Independent state-vector, signed-axis-range and dense mixed-QFIM checks."""
import argparse
import csv
from datetime import datetime, timezone
from itertools import product
import json
import hashlib
from pathlib import Path
import numpy as np
from core import (ROOT, catalogue, unsigned_profiles, aligned_signs, aggregate_profile,
                  block_signature, exact_score, AXES, GATES)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
H = (X+Z)/np.sqrt(2)
S = np.diag([1, 1j])
P = [X, Y, Z]
C = {"I": I, "X": X, "H": H, "HZ": H@Z, "HSdg": H@S.conj().T, "HS": H@S}


def tensor(operators):
    out = np.ones((1, 1), complex)
    for op in operators:
        out = np.kron(out, op)
    return out


def graph_state(n, edges):
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    phase = np.zeros(2**n, dtype=int)
    for i, j in edges:
        phase ^= bits[:, i]*bits[:, j]
    return ((-1.)**phase/np.sqrt(2**n)).astype(complex)


def dense_pure_axis_qfims(n, edges, choices):
    """All unsigned no-loss QFIMs from state vectors, NOT stabilizers."""
    psi = graph_state(n, edges)
    vectors = np.array([[tensor([P[a] if k == i else I for k in range(n)])@psi
                         for a in range(3)] for i in range(n)])
    means = np.einsum("d,iad->ia", psi.conj(), vectors).real
    table = np.einsum("iad,jbd->iajb", vectors.conj(), vectors).real-np.einsum("ia,jb->iajb", means, means)
    out = np.zeros((len(choices), n, n))
    for i in range(n):
        for j in range(n):
            out[:, i, j] = table[i, choices[:, i], j, choices[:, j]]
    return out


def verify_all_signs(n, seed=20261021):
    """Check every 6**n signed assignment on every LC graph representative.

    No-loss mean identifiability is independently tested by an eigenbasis range
    test. All losses are checked to introduce no pair correlation outside the
    no-loss signed blocks; hence all retained sign witnesses have identical loss
    QFIMs up to the proven per-block global sign invariance.
    """
    all_signs = np.array(list(product((1, -1), repeat=n)), dtype=np.int8)
    checked, valid_count, max_pure_error = 0, 0, 0.
    rng = np.random.default_rng(seed+n)
    max_random_mixed_error, random_mixed_count = 0., 0
    for item in catalogue(n):
        choices, profiles = unsigned_profiles(n, item["edges"])
        reference = dense_pure_axis_qfims(n, item["edges"], choices)
        error = float(np.max(np.abs(reference-profiles[:, -1])))
        max_pure_error = max(max_pure_error, error)
        if error > 1e-11:
            raise ArithmeticError("Stabilizer no-loss QFIM differs from state-vector reference")
        selected_axis = int(rng.integers(len(choices)))
        selected_signs = rng.choice([-1, 1], n)
        axes = [("+" if s > 0 else "-")+AXES[a] for a, s in zip(choices[selected_axis], selected_signs)]
        random_protocol = {"n": n, "edges": item["edges"], "sensor_gates": [GATES[a] for a in axes]}
        masks = sorted({(1 << n)-1, (1 << n)-1-(1 << int(rng.integers(n))), int(rng.integers(1, 1 << n))})
        mixed = dense_profile(random_protocol, masks=masks)
        expected = profiles[selected_axis]*selected_signs[None, :, None]*selected_signs[None, None, :]
        mixed_error = float(np.max(np.abs(mixed[masks]-expected[masks])))
        if mixed_error > 1e-10:
            raise ArithmeticError("Random signed-state mixed-QFIM mismatch")
        max_random_mixed_error = max(max_random_mixed_error, mixed_error)
        random_mixed_count += len(masks)
        # Independent range test, including every sign. Rounding restores exact
        # Pauli expectations after the state-vector arithmetic above.
        vals, vectors = np.linalg.eigh(np.rint(reference))
        coordinates = np.einsum("bji,sj->bsi", vectors, all_signs.astype(float)/n)
        positive = vals > 1e-9
        null_norm = np.linalg.norm(np.where(positive[:, None, :], 0, coordinates), axis=-1)
        identifiable = null_norm < 1e-8
        for b, profile in enumerate(profiles):
            signs, blocks = aligned_signs(profile[-1])
            predicted = np.zeros(len(all_signs), dtype=bool)
            if signs is not None:
                relative = all_signs*signs
                predicted[:] = True
                for block in blocks:
                    predicted &= np.all(relative[:, block] == relative[:, block[0], None], axis=1)
                aligned = profile*signs[None, :, None]*signs[None, None, :]
                if np.any(aligned < 0):
                    raise ArithmeticError("New negative loss correlation")
                if np.any((profile != 0) & (profile[-1] == 0)[None, :, :]):
                    raise ArithmeticError("Loss created a new pair correlation")
                block_signature(aggregate_profile(aligned), blocks)
                # A direct array comparison with one different valid sign
                # witness checks the phase/sign implementation at every loss.
                valid_signs = all_signs[np.flatnonzero(predicted)]
                alternate = valid_signs[-1]
                test = profile*alternate[None, :, None]*alternate[None, None, :]
                if not np.array_equal(test, aligned):
                    raise ArithmeticError("Equivalent sign witnesses differ after loss")
            if not np.array_equal(predicted, identifiable[b]):
                raise ArithmeticError("Sign reduction discarded an identifiable mean")
            checked += len(all_signs)
            valid_count += int(predicted.sum())
        print(json.dumps({"stage": "sign_verification", "n": n, "checked": checked}), flush=True)
    return {"signed_assignments_checked": checked, "identifiable_signed_assignments": valid_count,
            "maximum_state_vector_QFIM_error": max_pure_error,
            "random_signed_mixed_cases": random_mixed_count,
            "maximum_random_signed_mixed_QFIM_error": max_random_mixed_error,
            "all_signs_mean_range_checked": True, "all_unsigned_loss_supports_checked": True}


def dense_profile(protocol, phase_vector=None, masks=None):
    n = protocol["n"]
    psi = tensor([C[g] for g in protocol["sensor_gates"]])@graph_state(n, protocol["edges"])
    if phase_vector is not None:
        bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
        psi *= np.exp(-.5j*((1-2*bits)@np.array(phase_vector)))
    profile = np.zeros((1 << n, n, n))
    for mask in (range(1, 1 << n) if masks is None else masks):
        if mask == 0:
            continue
        kept = [i for i in range(n) if mask >> i & 1]
        lost = [i for i in range(n) if not mask >> i & 1]
        amplitude = psi.reshape([2]*n).transpose(kept+lost).reshape(2**len(kept), -1)
        rho = amplitude@amplitude.conj().T
        values, vectors = np.linalg.eigh(rho)
        values = np.clip(values, 0, None)
        denom = values[:, None]+values[None, :]
        weight = np.divide(2*(values[:, None]-values[None, :])**2, denom,
                           out=np.zeros_like(denom), where=denom > 1e-12)
        generators = [tensor([Z/2 if j == i else I for j in range(len(kept))]) for i in range(len(kept))]
        transformed = np.array([vectors.conj().T@g@vectors for g in generators])
        q = np.einsum("iab,jab,ab->ij", transformed, transformed.conj(), weight).real
        profile[mask][np.ix_(kept, kept)] = q
    return profile


def compare_runs(results, reference):
    own = {(r["n"], r["eta"]): r["effective_information_exact"]
           for r in csv.DictReader((results/"best_by_n_eta.csv").open())}
    matched = []
    for row in csv.DictReader((reference/"best_by_n_eta.csv").open()):
        key = row["n"], row["eta"]
        if key in own:
            if own[key] != row["effective_information_exact"]:
                raise ArithmeticError(f"Result mismatch at {key}")
            matched.append({"n": int(key[0]), "eta": key[1], "effective_information_exact": own[key]})
    if not matched:
        raise ValueError("No overlapping reference points")
    return {"matched_rows": matched, "all_matched": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT/"results")
    p.add_argument("--reference-results", type=Path)
    p.add_argument("--small-results", type=Path)
    p.add_argument("--skip-sign-enumeration", action="store_true", help="Quick saved-winner check only; recorded in report")
    p.add_argument("--seed", type=int, default=20261021)
    args = p.parse_args()
    summary = json.loads((args.results/"summary.json").read_text())
    for filename, digest in summary["source_sha256"].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != digest:
            raise ArithmeticError("Source differs from recorded run; rerun search before verification")
    protocols = json.loads((args.results/"protocols.json").read_text())
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(), "sign_checks": {},
              "saved_protocols_dense_checked": 0, "survivor_patterns_dense_checked": 0,
              "maximum_dense_QFIM_error": 0., "all_40_grid_values_checked_with_rational_arithmetic": False}
    report["maximum_dense_aggregate_error"] = 0.
    report["verification_seed"] = args.seed
    for n in range(summary["configuration"]["min_sensors"], summary["configuration"]["max_sensors"]+1):
        if not args.skip_sign_enumeration:
            report["sign_checks"][str(n)] = verify_all_signs(n, args.seed)
    for protocol in protocols.values():
        n = protocol["n"]
        profile = dense_profile(protocol)
        _, kernels = unsigned_profiles(n, protocol["edges"])
        choices = [AXES.index(a[1]) for a in protocol["effective_axes"]]
        signs = np.array([1 if a[0] == "+" else -1 for a in protocol["effective_axes"]])
        index = sum(a*3**(n-1-i) for i, a in enumerate(choices))
        expected = kernels[index]*signs[None, :, None]*signs[None, None, :]
        pattern_error = float(np.max(np.abs(profile-expected)))
        if pattern_error > 1e-10:
            raise ArithmeticError("Individual survivor-pattern physical QFIM mismatch")
        # Compare raw floating sums, so no discrepancy is hidden by casting.
        counts = np.array([s.bit_count() for s in range(1 << n)])
        dense_aggregate = np.stack([profile[counts == k].sum(axis=0) for k in range(n+1)])
        error = float(np.max(np.abs(dense_aggregate-np.array(protocol["A_by_survivor_count"]))))
        if error > 1e-10:
            raise ArithmeticError("Physical dense-state validation failed")
        report["maximum_dense_QFIM_error"] = max(report["maximum_dense_QFIM_error"], pattern_error)
        report["maximum_dense_aggregate_error"] = max(report["maximum_dense_aggregate_error"], error)
        report["saved_protocols_dense_checked"] += 1
        report["survivor_patterns_dense_checked"] += (1 << n)
    # Check nonzero, unequal independent phases as well (QFIM phase independence).
    phase_protocol = max(protocols.values(), key=lambda protocol: protocol["n"])
    phi = np.arange(phase_protocol["n"])*.137+.219
    phase_profile = dense_profile(phase_protocol, phi)
    counts = np.array([s.bit_count() for s in range(1 << phase_protocol["n"])])
    phase_aggregate = np.stack([phase_profile[counts == k].sum(axis=0) for k in range(phase_protocol["n"]+1)])
    phase_error = float(np.max(np.abs(phase_aggregate-np.array(phase_protocol["A_by_survivor_count"]))))
    if phase_error > 1e-10:
        raise ArithmeticError("Independent phase-vector verification failed")
    report["unequal_nonzero_phase_vector_error"] = phase_error
    rows = list(csv.DictReader((args.results/"best_by_n_eta.csv").open()))
    for row in rows:
        score = exact_score(protocols[row["protocol_id"]]["signature"], row["eta"])
        if str(score) != row["information_per_photon_exact"]:
            raise ArithmeticError("Rational saved-value mismatch")
    report["rational_grid_rows_checked"] = len(rows)
    report.pop("all_40_grid_values_checked_with_rational_arithmetic")
    if args.reference_results:
        report["graph_symmetry_reduction_comparison"] = compare_runs(args.results, args.reference_results)
    if args.small_results:
        report["previous_N1_to_5_comparison"] = compare_runs(args.small_results, ROOT.parent/"fixed_clifford_sensors/results")
    (args.results/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"stage": "verification_complete", "dense_protocols": report["saved_protocols_dense_checked"],
                      "max_dense_error": report["maximum_dense_QFIM_error"],
                      "sign_assignments": sum(x["signed_assignments_checked"] for x in report["sign_checks"].values())}), flush=True)


if __name__ == "__main__":
    main()
