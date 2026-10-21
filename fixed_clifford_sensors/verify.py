"""Independent dense-QFIM, complete small-N reduction, and saved-result checks."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
from itertools import product
import json
from pathlib import Path
import unittest
import numpy as np
from engine import (AXES, GATES, graph_catalogue, edges_from_mask, unsigned_profiles,
                    signed_profiles, effective_information, loss_weights, exact_information)
from run_search import eta_grid, parser as search_parser

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
    return ((-1.)**phase / np.sqrt(2**n)).astype(complex)


def reduced_state(psi, n, survivors):
    kept = [i for i in range(n) if survivors >> i & 1]
    lost = [i for i in range(n) if not survivors >> i & 1]
    amplitude = psi.reshape([2]*n).transpose(kept+lost).reshape(2**len(kept), -1)
    return kept, amplitude @ amplitude.conj().T


def dense_qfim(rho, generators):
    vals, vectors = np.linalg.eigh(rho)
    vals = np.clip(vals, 0, None)
    denom = vals[:, None]+vals[None, :]
    weight = np.divide(2*(vals[:, None]-vals[None, :])**2, denom,
                       out=np.zeros_like(denom), where=denom > 1e-12)
    transformed = np.array([vectors.conj().T @ g @ vectors for g in generators])
    return np.einsum("iab,jab,ab->ij", transformed, transformed.conj(), weight).real


def dense_profile(n, edges, axes, physical=False):
    psi = graph_state(n, edges)
    if physical:
        psi = tensor([C[GATES[a]] for a in axes]) @ psi
    out = np.zeros((1 << n, n, n))
    for survivors in range(1, 1 << n):
        kept, rho = reduced_state(psi, n, survivors)
        generators = []
        for k, i in enumerate(kept):
            sign = 1 if axes[i][0] == "+" else -1
            op = Z if physical else sign*P[AXES.index(axes[i][1])]
            generators.append(tensor([op/2 if j == k else I for j in range(len(kept))]))
        q = dense_qfim(rho, generators)
        out[survivors][np.ix_(kept, kept)] = q
    return out


class MathematicalChecks(unittest.TestCase):
    def test_catalogue_and_candidate_counts(self):
        self.assertEqual([len(graph_catalogue(n)) for n in range(1, 6)], [1, 2, 3, 6, 11])
        self.assertEqual(sum(len(graph_catalogue(n))*6**n for n in range(1, 6)), 94038)
        for n in range(1, 6):
            self.assertEqual(sum(len(c["unlabeled_members"]) for c in graph_catalogue(n)), [1,2,4,11,34][n-1])

    def test_gate_table(self):
        for a, gate in GATES.items():
            expected = (1 if a[0] == "+" else -1)*P[AXES.index(a[1])]
            np.testing.assert_allclose(C[gate].conj().T@Z@C[gate], expected, atol=1e-14)

    def test_all_three_sensor_unsigned_profiles_against_dense(self):
        n = 3
        for g in range(1 << (n*(n-1)//2)):
            edges = edges_from_mask(n, g)
            choices, profiles = unsigned_profiles(n, edges)
            for indices, expected in zip(choices, profiles):
                axes = ["+"+AXES[i] for i in indices]
                np.testing.assert_allclose(dense_profile(n, edges, axes), expected, atol=1e-12)

    def test_four_five_sensor_physical_rotations_and_loss(self):
        rng = np.random.default_rng(20261021)
        for n in (4, 5):
            for item in graph_catalogue(n):
                choices, profiles = unsigned_profiles(n, item["edges"])
                for _ in range(3):
                    k = int(rng.integers(len(choices)))
                    signs = rng.choice([-1, 1], n)
                    axes = [("+" if s > 0 else "-")+AXES[a] for s, a in zip(signs, choices[k])]
                    expected = profiles[k]*signs[None, :, None]*signs[None, None, :]
                    np.testing.assert_allclose(dense_profile(n, item["edges"], axes, physical=True), expected, atol=1e-12)

    def test_loss_and_identifiability_benchmarks(self):
        etas = np.array([0., 0.05, 0.3, 0.8, 1.])
        for n in range(1, 6):
            probs = loss_weights(n, etas)
            np.testing.assert_allclose(probs.sum(axis=1), 1)
            independent = dense_profile(n, [], ["+Z"]*n, physical=True)
            fi = effective_information(np.einsum("es,sij->eij", probs, independent))
            np.testing.assert_allclose(fi, n*etas, atol=1e-12)
            star = [[0, j] for j in range(1, n)]
            axes = ["+Z"]+["+X"]*(n-1)
            ghz = dense_profile(n, star, axes, physical=True)
            fi = effective_information(np.einsum("es,sij->eij", probs, ghz))
            np.testing.assert_allclose(fi, n*n*etas**n, atol=1e-12)
        self.assertEqual(float(effective_information(np.diag([1., 0.]))), 0)
        self.assertEqual(float(effective_information(np.array([[1., -1.], [-1., 1.]]))), 0)
        self.assertAlmostEqual(float(effective_information(np.ones((2, 2)))), 4.)

    def test_reduction_preserves_all_labeled_profiles_up_to_n3(self):
        for n in range(1, 4):
            reduced = {p.tobytes() for c in graph_catalogue(n) for _, p in signed_profiles(n, c["edges"])}
            full = {p.tobytes() for g in range(1 << (n*(n-1)//2)) for _, p in signed_profiles(n, edges_from_mask(n, g))}
            # Label permutations must be restored; compare sorted profile invariants
            # by brute-force relabeling the reduced profiles, INCLUDING survivors.
            from itertools import permutations
            restored = set()
            for blob in reduced:
                profile = np.frombuffer(blob, dtype=np.int8).reshape(1 << n, n, n)
                for perm in permutations(range(n)):
                    transformed = np.zeros_like(profile)
                    for oldmask in range(1 << n):
                        newmask = sum(1 << perm[i] for i in range(n) if oldmask >> i & 1)
                        transformed[newmask][np.ix_(perm, perm)] = profile[oldmask]
                    restored.add(transformed.tobytes())
            self.assertEqual(restored, full)

    def test_eta_configuration(self):
        p = search_parser()
        self.assertEqual(eta_grid(p.parse_args([])), [str(Decimal(i)/20) for i in range(1, 21)])
        self.assertEqual(eta_grid(p.parse_args(["--etas", "1,0,0.25,0.25"])), ["0", "0.25", "1"])
        with self.assertRaises(ValueError):
            eta_grid(p.parse_args(["--etas", "-0.1"]))


def verify_results(results):
    import csv
    summary = json.loads((results / "summary.json").read_text())
    protocols = json.loads((results / "protocols.json").read_text())
    maximum_error, checked = 0., 0
    for protocol in protocols.values():
        dense = dense_profile(protocol["n"], protocol["edges"], protocol["effective_axes"], physical=True)
        error = float(np.max(np.abs(dense-np.array(protocol["qfi_by_survivor_mask"]))))
        maximum_error = max(maximum_error, error)
        if error > 1e-10:
            raise ArithmeticError("Saved protocol differs from dense calculation")
        checked += 1
    rows = list(csv.DictReader((results / "best_by_n_eta.csv").open()))
    for row in rows:
        profile = protocols[row["protocol_id"]]["qfi_by_survivor_mask"]
        exact, _ = exact_information(profile, row["eta"])
        if str(exact) != row["effective_information_exact"]:
            raise ArithmeticError("Saved rational information mismatch")
    return {"verified_utc": datetime.now(timezone.utc).isoformat(),
            "saved_protocols_dense_checked": checked, "saved_grid_rows_exact_checked": len(rows),
            "maximum_dense_QFIM_error": maximum_error,
            "candidate_count": sum(s["evaluated_candidates"] for s in summary["search_statistics"])}


def compare_unreduced(results, comparison):
    import csv
    a = {(r["n"], r["eta"]): r["effective_information_exact"]
         for r in csv.DictReader((results/"best_by_n_eta.csv").open())}
    other_summary = json.loads((comparison/"summary.json").read_text())
    if other_summary["configuration"]["graph_reduction"] != "labeled":
        raise ValueError("Comparison run must use --graph-reduction labeled")
    matched = []
    for row in csv.DictReader((comparison/"best_by_n_eta.csv").open()):
        key = row["n"], row["eta"]
        if key in a:
            if a[key] != row["effective_information_exact"]:
                raise ArithmeticError(f"Unreduced comparison mismatch at {key}")
            matched.append({"n": int(key[0]), "eta": key[1], "effective_information_exact": a[key]})
    if not matched:
        raise ValueError("No overlapping grid points to compare")
    portable_configuration = dict(other_summary["configuration"])
    portable_configuration["output"] = comparison.name
    return {"compared_configuration": portable_configuration,
            "compared_candidates": sum(s["evaluated_candidates"] for s in other_summary["search_statistics"]),
            "matched_rows": matched, "all_matched": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--unreduced-results", type=Path, help="Optional labeled-graph run to compare")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MathematicalChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if not args.tests_only:
        report = verify_results(args.results)
        report.update({"mathematical_tests_run": result.testsRun, "mathematical_tests_passed": True})
        if args.unreduced_results:
            report["unreduced_comparison"] = compare_unreduced(args.results, args.unreduced_results)
        (args.results/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
