"""Exhaustive stabilizer-reduced search for eight sensors, parameterized CLI."""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from core import (ROOT, catalogue, axis_orbits, unsigned_profiles, aligned_signs,
                  aggregate_profile, block_signature, score_signature, exact_score,
                  protocol, integer_partitions, ghz_partition_score,
                  signature_dominance_bound, effective_information)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-sensors", type=int, default=8)
    p.add_argument("--max-sensors", type=int, default=8)
    p.add_argument("--eta-min", default="0.05")
    p.add_argument("--eta-max", default="1")
    p.add_argument("--eta-points", type=int, default=20)
    p.add_argument("--etas", help="Comma-separated probabilities; overrides uniform grid")
    p.add_argument("--axis-reduction", choices=("symmetry", "none"), default="symmetry")
    p.add_argument("--output", type=Path, default=ROOT/"results")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tie-tol", type=float, default=1e-10)
    p.add_argument("--log-every-graphs", type=int, default=1, help="0 disables progress")
    return p


def eta_grid(args):
    if args.etas:
        values = [Decimal(x.strip()) for x in args.etas.split(",")]
    else:
        lo, hi = Decimal(args.eta_min), Decimal(args.eta_max)
        if args.eta_points < 1 or lo > hi:
            raise ValueError("Invalid grid")
        values = [lo] if args.eta_points == 1 else [lo+(hi-lo)*k/(args.eta_points-1) for k in range(args.eta_points)]
    if not values or any(not x.is_finite() or not 0 <= x <= 1 for x in values):
        raise ValueError("Require finite eta in [0,1]")
    return [str(x.normalize()) for x in sorted(set(values))]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = arguments().parse_args(argv)
    if not 1 <= args.min_sensors <= args.max_sensors <= 8:
        raise ValueError("Supported sensor counts: 1..8")
    if not np.isfinite(args.tie_tol) or args.tie_tol <= 0 or args.log_every_graphs < 0:
        raise ValueError("Invalid tolerance or log interval")
    grid = eta_grid(args)
    etas = np.array([float(x) for x in grid])
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Nonempty output; choose a new --output or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    source_files = sorted(ROOT.glob("*.py"))+[ROOT/"requirements.txt", ROOT.parent/"fixed_clifford_sensors/engine.py"]
    source_hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else "../fixed_clifford_sensors/engine.py":
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    start, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    best_rows, graph_rows, stat_rows, winners = [], [], [], {}
    for n in range(args.min_sensors, args.max_sensors+1):
        nstart = time.perf_counter()
        graphs, candidates, unsigned_count = catalogue(n), 0, 0
        records, seen, all_scores, graph_stats = [], {}, [], []
        max_bound_violation = 0.
        max_eigensolver_error = 0.
        for graph_id, item in enumerate(graphs):
            orbits, unsigned_raw, theoretical, aut_count = axis_orbits(item["adjacent"], args.axis_reduction)
            unsigned_count += unsigned_raw
            # One graph-sized temporary batch; at N=8 this is ~107 MB of int8,
            # never an array for all graphs/candidates. Retained records use A_k.
            choices_all, profiles = unsigned_profiles(n, item["edges"])
            per_graph_best = np.zeros(len(grid))
            for choices, multiplicity in orbits:
                candidates += 1
                choice_index = 0
                for a in choices:
                    choice_index = 3*choice_index+a
                raw = profiles[choice_index]
                signs, blocks = aligned_signs(raw[-1])
                if signs is None:
                    raise ArithmeticError("Unexpected unidentifiable axes after isolated-sensor filter")
                aligned = raw*signs[None, :, None]*signs[None, None, :]
                if np.any(aligned < 0):
                    raise ArithmeticError("Loss generated a new or frustrated pair correlation")
                aggregate = aggregate_profile(aligned)
                signature = block_signature(aggregate, blocks)
                if signature not in seen:
                    scores = score_signature(signature, etas)
                    k = np.arange(n+1)
                    coefficients = etas[:, None]**k*(1-etas[:, None])**(n-k)
                    matrices = np.einsum("ek,kij->eij", coefficients, aggregate)
                    eigen_scores = effective_information(matrices)/n
                    max_eigensolver_error = max(max_eigensolver_error, float(np.max(np.abs(scores-eigen_scores))))
                    if not np.allclose(scores, eigen_scores, rtol=1e-8, atol=1e-10):
                        raise ArithmeticError("Block score differs from direct QFIM eigensolver")
                    bound = signature_dominance_bound(signature, grid)
                    max_bound_violation = max(max_bound_violation, float(np.max(scores-bound)))
                    if np.any(scores > bound+1e-9):
                        raise ArithmeticError("GHZ-block dominance bound violated")
                    seen[signature] = len(records)
                    records.append(protocol(n, graph_id, item, choices, signs, blocks, aggregate, signature))
                    all_scores.append(scores)
                else:
                    scores = all_scores[seen[signature]]
                per_graph_best = np.maximum(per_graph_best, scores)
            graph_stats.append({"graph_id": graph_id, "components": item["components"], "edges": item["edges"],
                                "automorphisms": aut_count, "valid_unsigned_axes": unsigned_raw,
                                "evaluated_axis_orbits": len(orbits), "burnside_orbits": theoretical})
            for e, eta in enumerate(grid):
                graph_rows.append({"n": n, "graph_id": graph_id, "eta": eta,
                                   "best_information_per_photon": float(per_graph_best[e])})
            if args.log_every_graphs and (graph_id+1) % args.log_every_graphs == 0:
                print(json.dumps({"stage": "graph", "n": n, "graph": graph_id+1,
                                  "of": len(graphs), "axis_candidates": candidates,
                                  "unique_QFIM_polynomials": len(records)}), flush=True)
        all_scores = np.array(all_scores)
        partitions = list(integer_partitions(n))
        for e, eta in enumerate(grid):
            numeric_best = float(np.max(all_scores[:, e]))
            indices = np.flatnonzero(all_scores[:, e] >= numeric_best-args.tie_tol*max(1, numeric_best))
            exact_values = [(exact_score(records[i]["signature"], eta), int(i)) for i in indices]
            best = max(v for v, _ in exact_values)
            chosen = next(i for value, i in exact_values if value == best)
            row_protocol = records[chosen]
            protocol_id = f"N{n}_g{row_protocol['graph_id']}_"+"_".join(a.replace("+", "p").replace("-", "m") for a in row_protocol["effective_axes"])
            winners[protocol_id] = row_protocol
            baseline = Fraction(eta)
            partition_values = [(ghz_partition_score(p, eta), p) for p in partitions]
            best_partition_value = max(v for v, _ in partition_values)
            best_partition = next(p for v, p in partition_values if v == best_partition_value)
            # This comparison is an independent benchmark, not a search restriction.
            if best != best_partition_value:
                raise ArithmeticError("Exhaustive winner differs from GHZ-partition upper benchmark")
            best_rows.append({"n": n, "eta": eta, "protocol_id": protocol_id,
                              "information_per_photon": float(best), "information_per_photon_exact": str(best),
                              "effective_information": float(n*best), "effective_information_exact": str(n*best),
                              "mean_QCRB_per_trial": float(1/(n*best)) if best else "inf",
                              "independent_per_photon": float(baseline),
                              "ghz_per_photon": float(n*baseline**n),
                              "gain_over_independent": float(best/baseline) if baseline else "",
                              "best_GHZ_partition": "+".join(map(str, best_partition)),
                              "partition_bound_exact": str(best_partition_value),
                              "tied_QFIM_polynomials_among_contenders": sum(v == best for v, _ in exact_values)})
        expected = {1: 2, 2: 9, 3: 34, 4: 155, 5: 792, 6: 4928, 7: 38344}
        if args.axis_reduction == "symmetry" and n in expected and candidates != expected[n]:
            raise ArithmeticError("Axis-orbit total count mismatch")
        stat_rows.append({"n": n, "lc_representatives": len(graphs), "unreduced_signed_axes": len(graphs)*6**n,
                          "valid_unsigned_axes": unsigned_count, "evaluated_axis_candidates": candidates,
                          "unique_QFIM_polynomials": len(records), "seconds": time.perf_counter()-nstart,
                          "max_GHZ_dominance_bound_violation": max_bound_violation,
                          "all_unique_polynomials_checked_with_QFIM_eigensolver": len(records),
                          "max_block_score_eigensolver_error": max_eigensolver_error,
                          "graph_statistics": graph_stats})
        print(json.dumps({"stage": "completed_n", "n": n, "candidates": candidates,
                          "polynomials": len(records), "seconds": stat_rows[-1]["seconds"]}), flush=True)
    summary = {"configuration": {**vars(args), "output": output.name, "eta_grid": grid,
                                  "ancillas": 0, "target": "arithmetic_mean", "fixed_before_loss": True,
                                  "phase_generator": "Z/2", "loss": "independent_flagged_erasure",
                                  "phase_location": "after_loss", "readout": "QFI only"},
               "search_statistics": stat_rows, "started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter()-start,
               "source_sha256": source_hashes,
               "environment": {"python": platform.python_version(), "numpy": np.__version__},
               "optimality": "Exhaustive fixed-Clifford search modulo exact stabilizer/sign/graph reductions; numerical screening with rational contenders",
               "tie_policy": "one witness per information-polynomial signature, first rational maximizer retained"}
    write_csv(output/"best_by_n_eta.csv", best_rows)
    write_csv(output/"best_by_graph.csv", graph_rows)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    (output/"protocols.json").write_text(json.dumps(winners, ensure_ascii=False, separators=(",", ":"))+"\n")
    print(json.dumps({"stage": "finished", "seconds": summary["elapsed_seconds"], "output": output.name}), flush=True)


if __name__ == "__main__":
    main()
