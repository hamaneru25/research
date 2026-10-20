"""Deduplicate graph-state QFIMs and optimize randomized sensing by SDP.

Model: three sensor qubits, three retained central qubits; heralded independent
erasure BEFORE the unknown local Z phases. Central bases are X/Y; all local
single-qubit Cliffords and recorded randomized choices are allowed. No trial
is postselected. All six prepared qubits count toward the resource budget.

Run: python optimize_graph_sensing.py --output results
Requires numpy, scipy, cvxpy and the accompanying graph_sensing_core.py.
"""
import argparse
import csv
import gzip
import json
import time
from itertools import combinations, permutations, product
from pathlib import Path

import cvxpy as cp
import numpy as np

from graph_sensing_core import (
    H, I, PAULI, X, Z, conditional_sensor_states, effective_information,
    graph_state, local_generators, qufim, tensor, validate_core,
)

EDGES = list(combinations(range(6), 2))
BASES = list(product("XY", repeat=3))
SURVIVORS = [s for k in range(1, 4) for s in combinations(range(3), k)]
W = np.ones(3) / 3


def graph_representatives():
    """All simple graphs modulo S3(sensor) x S3(central), not LC equivalence."""
    masks = np.arange(1 << 15, dtype=np.uint16)
    edge_index = {e: i for i, e in enumerate(EDGES)}
    transformed = []
    for ps in permutations(range(3)):
        for pc in permutations(range(3, 6)):
            perm = ps + pc
            out = np.zeros_like(masks)
            for j, (u, v) in enumerate(EDGES):
                dst = edge_index[tuple(sorted((perm[u], perm[v])))]
                out |= ((masks >> j) & 1) << dst
            transformed.append(out)
    transformed = np.asarray(transformed)
    visited = np.zeros(1 << 15, dtype=bool)
    representatives = []
    for mask in range(1 << 15):
        if not visited[mask]:
            representatives.append(mask)
            visited[transformed[:, mask]] = True
    assert visited.all() and len(representatives) == 1408
    return representatives


def edges_of(mask):
    return [e for j, e in enumerate(EDGES) if (mask >> j) & 1]


def matrix_key(matrix):
    """These stabilizer/Pauli QFIM entries are integer; verify before rounding."""
    rounded = np.rint(matrix).astype(np.int8)
    if np.max(np.abs(matrix - rounded)) > 1e-8:
        raise ArithmeticError("A QFIM was not integer within numerical precision")
    return tuple(int(x) for x in rounded.ravel())


class CandidateFactory:
    def __init__(self):
        self.cache = {}
        self.generators = {}
        self.choices = {}
        self.indices = {}
        self.signs = {}
        for k in (1, 2, 3):
            self.generators[k] = tuple(
                tensor([PAULI[a] / 2 if j == i else I for j in range(k)])
                for i in range(k) for a in "XYZ"
            )
            entries = [(a, (1,) + s) for a in product(range(3), repeat=k)
                       for s in product((1, -1), repeat=k-1)]
            self.choices[k] = entries
            self.indices[k] = np.array([[3*i + a[i] for i in range(k)]
                                        for a, s in entries])
            self.signs[k] = np.array([s for a, s in entries])

    def candidates(self, rho):
        k = int(round(np.log2(len(rho))))
        scale = 2 ** k
        real = np.rint(rho.real * scale).astype(np.int8)
        imag = np.rint(rho.imag * scale).astype(np.int8)
        if max(np.max(abs(rho.real-real/scale)), np.max(abs(rho.imag-imag/scale))) > 1e-8:
            raise ArithmeticError("State not on the expected stabilizer grid")
        key = (k, real.tobytes(), imag.tobytes())
        if key not in self.cache:
            full = qufim(rho, self.generators[k])
            ids = self.indices[k]
            signs = self.signs[k]
            qs = full[ids[:, :, None], ids[:, None, :]] * signs[:, :, None] * signs[:, None, :]
            unique = {}
            for q, (axes, sign) in zip(qs, self.choices[k]):
                unique.setdefault(matrix_key(q), (axes, sign))
            self.cache[key] = unique
        return self.cache[key]


def build_catalog(output):
    """Union QFIMs across bases, after verifying equal outcome reachable sets.

    For a fixed S,b, every nonzero outcome must have the same reachable set.
    Then each listed QFIM can be realized for EVERY outcome via feed-forward.
    Randomizing between those strategies realizes their convex hull. Taking
    the union over b therefore preserves the full randomized feasible region.
    """
    cache_path = output / "candidate_catalog.json.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt") as f:
            return json.load(f)
    factory = CandidateFactory()
    graphs = []
    profiles = []
    profile_ids = {}
    branches = 0
    started = time.perf_counter()
    for gi, mask in enumerate(graph_representatives()):
        psi = graph_state(6, edges_of(mask))
        loss_candidates = []
        for survivors in SURVIVORS:
            union = set()
            for bases in BASES:
                candidate_sets = []
                for signs, p, rho in conditional_sensor_states(psi, 3, 3, survivors, bases):
                    branches += 1
                    candidate_sets.append(frozenset(factory.candidates(rho)))
                if len(set(candidate_sets)) != 1:
                    raise ArithmeticError("Outcome compression is not valid for this block")
                union.update(candidate_sets[0])
            loss_candidates.append(tuple(sorted(union)))
        signature = tuple(loss_candidates)
        if signature not in profile_ids:
            profile_ids[signature] = len(profiles)
            profiles.append({"id": len(profiles), "first_graph_mask": mask,
                             "candidates": loss_candidates})
        graphs.append({"mask": mask, "edges": edges_of(mask),
                       "profile_id": profile_ids[signature]})
        if (gi + 1) % 256 == 0:
            print(json.dumps({"stage": "catalog", "graphs": gi+1,
                              "profiles": len(profiles)}, ensure_ascii=False), flush=True)
    catalog = {"model": {"sensors": [0, 1, 2], "central": [3, 4, 5],
                          "central_bases": "X/Y", "sensor_operations": "all local Clifford",
                          "loss_before_phase": True, "known_loss_pattern": True,
                          "phase_generator": "(I-Z_i)/2", "target_weights": W.tolist(),
                          "randomized_choices_recorded": True, "generated_photons": 6},
               "survivors": SURVIVORS, "graphs": graphs, "profiles": profiles,
               "audit": {"graph_count": len(graphs), "profile_count": len(profiles),
                         "conditional_blocks": 1408*7*8, "nonzero_branches": branches,
                         "outcome_reachable_sets_verified_equal": True,
                         "max_probability_variables": max(sum(map(len, p["candidates"])) for p in profiles)}}
    with gzip.open(cache_path, "wt") as f:
        json.dump(catalog, f)
    print(json.dumps({"stage": "catalog_complete", **catalog["audit"]}), flush=True)
    return catalog


class RandomizedSDP:
    def __init__(self, profile):
        self.profile = profile
        self.matrices = []
        self.loss_ids = []
        self.keys = []
        for loss_id, (survivors, candidates) in enumerate(zip(SURVIVORS, profile["candidates"])):
            for key in candidates:
                q = np.zeros((3, 3))
                q[np.ix_(survivors, survivors)] = np.array(key).reshape(len(survivors), -1)
                self.matrices.append(q)
                self.loss_ids.append(loss_id)
                self.keys.append(tuple(key))
        self.matrices = np.asarray(self.matrices)
        self.loss_ids = np.asarray(self.loss_ids)
        self.x = cp.Variable(len(self.keys))
        self.t = cp.Variable()
        self.coefficients = cp.Parameter(7, nonneg=True)
        design = self.matrices.reshape(len(self.keys), 9).T
        vector = design @ cp.multiply(self.coefficients[self.loss_ids], self.x)
        q = cp.reshape(vector, (3, 3), order="C")
        self.q_expr = (q + q.T) / 2
        groups = np.array([self.loss_ids == i for i in range(7)], float)
        self.constraints = [self.x >= 0, groups @ self.x == np.ones(7),
                            cp.bmat([[self.q_expr, W[:, None]],
                                     [W[None, :], cp.reshape(self.t, (1, 1), order="C")]]) >> 0]
        self.problem = cp.Problem(cp.Minimize(self.t), self.constraints)
        assert self.problem.is_dcp()

    def solve(self, eta):
        # Scale by eta to improve conditioning without changing the optimizer.
        coefficients = np.array([eta**(len(s)-1)*(1-eta)**(3-len(s)) for s in SURVIVORS])
        self.coefficients.value = coefficients
        last = None
        for precision in (1e-8, 1e-10):
            self.problem.solve(solver="CLARABEL", warm_start=True, max_iter=300,
                               tol_gap_abs=precision, tol_gap_rel=precision,
                               tol_feas=precision)
            if self.problem.status not in ("optimal", "optimal_inaccurate"):
                raise ArithmeticError(f"SDP failed: {self.problem.status}")
            raw = self.x.value.copy()
            x = np.maximum(raw, 0)
            for j in range(7):
                ids = self.loss_ids == j
                x[ids] /= x[ids].sum()
            fbar = np.einsum("a,aij->ij", x*coefficients[self.loss_ids], self.matrices)
            total = eta * fbar
            feasible_feff = effective_information(total)
            # Independent c-optimal design certificate. For ANY direction d,
            # min_design w'Fbar^+w >= (w'd)^2 / max_design d'Fbar d.
            # The max separates over the seven loss groups.
            directions = [np.linalg.pinv(fbar, rcond=1e-12) @ W, W]
            dual = self.constraints[-1].dual_value
            if dual is not None and dual[3, 3] > 0:
                directions.append(-dual[:3, 3] / dual[3, 3])
            variance_lower = 0.0
            for d in directions:
                scores = np.einsum("i,aij,j->a", d, self.matrices, d)
                maximum = sum(coefficients[j]*scores[self.loss_ids == j].max() for j in range(7))
                if maximum > 0:
                    variance_lower = max(variance_lower, float((W @ d)**2 / maximum))
            feff_upper = eta / variance_lower
            relative_gap = max(0.0, feff_upper-feasible_feff) / max(feasible_feff, 1e-15)
            psd_matrix = np.block([[fbar, W[:, None]], [W[None, :], np.array([[self.t.value]])]])
            last = {"eta": float(eta), "profile_id": self.profile["id"],
                    "feff": feasible_feff, "feff_upper": feff_upper,
                    "relative_gap": relative_gap, "feff_per_generated_photon": feasible_feff/6,
                    "status": self.problem.status, "qfim": total.tolist(), "weights": x.tolist(),
                    "probability_variables": len(x),
                    "raw_min_probability": float(raw.min()),
                    "psd_min_eigenvalue": float(np.linalg.eigvalsh(psd_matrix).min())}
            if relative_gap <= 2e-6:
                return last
        if last["relative_gap"] > 2e-5:
            raise ArithmeticError(f"Optimality gap too large: {last}")
        return last


def witness(mask, survivors, target_key, factory):
    psi = graph_state(6, edges_of(mask))
    for bases in BASES:
        outcomes = []
        for result, p, rho in conditional_sensor_states(psi, 3, 3, survivors, bases):
            candidates = factory.candidates(rho)
            if tuple(target_key) not in candidates:
                break
            axes, signs = candidates[tuple(target_key)]
            outcomes.append({"central_result": list(result), "probability": p,
                             "pulled_back_axes": ["XYZ"[a] for a in axes],
                             "axis_signs": list(signs)})
        else:
            return {"central_bases": list(bases), "outcomes": outcomes}
    raise ArithmeticError("No physical strategy witnesses the candidate matrix")


def physical_qfim(mask, survivors, strategy):
    sgate = np.diag([1, 1j])
    gates = {("X", 1): H, ("X", -1): H@Z,
             ("Y", 1): H@sgate.conj().T, ("Y", -1): H@sgate,
             ("Z", 1): I, ("Z", -1): X}
    expected = {tuple(o["central_result"]): o for o in strategy["outcomes"]}
    psi = graph_state(6, edges_of(mask))
    average = np.zeros((3, 3))
    for r, p, rho in conditional_sensor_states(psi, 3, 3, survivors, strategy["central_bases"]):
        branch = expected[r]
        gate = tensor([gates[(a, s)] for a, s in zip(branch["pulled_back_axes"], branch["axis_signs"])])
        actual = qufim(gate@rho@gate.conj().T, local_generators(len(survivors), ("Z",)*len(survivors)))
        average[np.ix_(survivors, survivors)] += p*actual
    return average


def optimize(catalog, etas, output):
    results = []
    for j, profile in enumerate(catalog["profiles"]):
        optimizer = RandomizedSDP(profile)
        for eta in etas:
            results.append(optimizer.solve(float(eta)))
        if (j+1) % 20 == 0 or j+1 == len(catalog["profiles"]):
            print(json.dumps({"stage": "sdp", "profiles_done": j+1,
                              "profiles_total": len(catalog["profiles"]),
                              "solves": len(results),
                              "max_relative_gap": max(r["relative_gap"] for r in results)}), flush=True)
    with gzip.open(output/"profile_sdp_results.json.gz", "wt") as f:
        json.dump(results, f)
    lookup = {(r["profile_id"], round(r["eta"], 10)): r for r in results}
    with open(output/"all_graph_results.csv", "w", newline="") as f:
        columns = ["eta", "graph_mask", "profile_id", "edge_count", "feff", "feff_upper",
                   "feff_per_generated_photon", "relative_gap"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for g in catalog["graphs"]:
            for eta in etas:
                r = lookup[g["profile_id"], round(float(eta), 10)]
                writer.writerow({"graph_mask": g["mask"], "edge_count": len(g["edges"]),
                                 **{key:r[key] for key in columns if key not in ("graph_mask", "edge_count")}})

    winners = []
    factory = CandidateFactory()
    for eta in etas:
        rows = [r for r in results if abs(r["eta"]-eta) < 1e-10]
        best = max(rows, key=lambda r:r["feff"])
        tolerance = 2e-6*max(1.0, best["feff"])
        near_profiles = {r["profile_id"] for r in rows if r["feff"] >= best["feff"]-tolerance}
        near_graphs = [g for g in catalog["graphs"] if g["profile_id"] in near_profiles]
        representative = min(near_graphs, key=lambda g:(len(g["edges"]), g["mask"]))
        chosen = lookup[representative["profile_id"], round(float(eta), 10)]
        profile = catalog["profiles"][representative["profile_id"]]
        optimizer = RandomizedSDP(profile)
        weights = np.asarray(chosen["weights"])
        policies = []
        reconstructed = np.zeros((3, 3))
        maximum_matrix_error = 0.0
        for loss_id, survivors in enumerate(SURVIVORS):
            actions = []
            p_loss = float(eta**len(survivors)*(1-eta)**(3-len(survivors)))
            for i in np.flatnonzero(optimizer.loss_ids == loss_id):
                # Preserve every weight, including small positive numerical weights.
                strategy = witness(representative["mask"], survivors, optimizer.keys[i], factory)
                actual = physical_qfim(representative["mask"], survivors, strategy)
                maximum_matrix_error = max(maximum_matrix_error, float(np.max(abs(actual-optimizer.matrices[i]))))
                reconstructed += p_loss*weights[i]*actual
                actions.append({"choice_probability": float(weights[i]),
                                "qfim": optimizer.matrices[i].tolist(), **strategy})
            policies.append({"survivors": list(survivors), "loss_probability": p_loss, "actions": actions})
        if not np.allclose(reconstructed, chosen["qfim"], atol=1e-8, rtol=1e-8):
            raise ArithmeticError("Exported physical protocol does not reproduce its QFIM")
        winners.append({"eta": float(eta), "best_feff": best["feff"],
                        "global_feff_upper": max(r["feff_upper"] for r in rows),
                        "representative_graph_mask": representative["mask"],
                        "representative_edges": representative["edges"],
                        "representative_feff": chosen["feff"],
                        "feff_per_generated_photon": chosen["feff"]/6,
                        "qfim": chosen["qfim"], "near_optimal_graph_count": len(near_graphs),
                        "near_optimal_tolerance": tolerance,
                        "near_optimal_graph_masks": [g["mask"] for g in near_graphs],
                        "max_witness_matrix_error": maximum_matrix_error,
                        "reconstructed_feff": effective_information(reconstructed),
                        "policies": policies})
    with open(output/"optimal_protocols.json", "w") as f:
        json.dump(winners, f, indent=2)
    with open(output/"optimal_summary.csv", "w", newline="") as f:
        cols = ["eta", "best_feff", "global_feff_upper", "representative_graph_mask",
                "representative_feff", "feff_per_generated_photon", "near_optimal_graph_count"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in winners:
            writer.writerow({key:row[key] for key in cols})
    return {"solves": len(results), "max_relative_gap": max(r["relative_gap"] for r in results),
            "solver_statuses": sorted(set(r["status"] for r in results)),
            "max_witness_error": max(r["max_witness_matrix_error"] for r in winners),
            "winners": [{key:r[key] for key in ("eta", "best_feff", "representative_graph_mask", "representative_edges")}
                        for r in winners]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--etas", type=str, default=",".join(str(i/20) for i in range(1, 21)))
    args = parser.parse_args()
    etas = np.array([float(x) for x in args.etas.split(",")])
    if np.any(etas <= 0) or np.any(etas > 1):
        raise ValueError("Use 0 < eta <= 1; eta=0 has zero information analytically")
    args.output.mkdir(parents=True, exist_ok=True)
    validate_core()
    catalog = build_catalog(args.output)
    exact_profile_ids = [p["id"] for p in catalog["profiles"]
                         if all([1]*(len(s)**2) in list(map(list, keys))
                                for s, keys in zip(SURVIVORS, p["candidates"]))]
    with open(args.output/"exact_optimal_graphs.json", "w") as f:
        json.dump({"criterion": "Every nonempty loss pattern admits QFIM ones(K,K)",
                   "profile_ids": exact_profile_ids,
                   "graphs": [g for g in catalog["graphs"] if g["profile_id"] in exact_profile_ids]},
                  f, indent=2)
    result = optimize(catalog, etas, args.output)
    with open(args.output/"validation.json", "w") as f:
        json.dump({"catalog": catalog["audit"], **result}, f, indent=2)
    print("FINAL " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
