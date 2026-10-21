"""Parameterised exhaustive graph + fixed local-Clifford search (no ancillas)."""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from engine import (search_catalogue, signed_profiles, loss_weights,
                    effective_information, exact_information, describe)

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-sensors", type=int, default=1)
    p.add_argument("--max-sensors", type=int, default=5)
    p.add_argument("--eta-min", default="0.05", help="Inclusive grid start")
    p.add_argument("--eta-max", default="1", help="Inclusive grid end")
    p.add_argument("--eta-points", type=int, default=20)
    p.add_argument("--etas", help="Explicit comma-separated survival probabilities; overrides grid")
    p.add_argument("--graph-reduction", choices=("lc", "isomorphism", "labeled"), default="lc",
                   help="lc: LC + relabeling; labeled: no graph reduction (much slower)")
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--overwrite", action="store_true", help="Replace this run's output files")
    p.add_argument("--eig-rtol", type=float, default=1e-10)
    p.add_argument("--range-tol", type=float, default=1e-8)
    p.add_argument("--tie-tol", type=float, default=1e-10)
    p.add_argument("--log-every", type=int, default=10000, help="Candidate progress interval; 0 disables")
    return p


def eta_grid(args):
    if args.etas:
        values = [Decimal(v.strip()) for v in args.etas.split(",")]
    else:
        low, high = Decimal(args.eta_min), Decimal(args.eta_max)
        if args.eta_points < 1 or low > high:
            raise ValueError("Require eta-points >= 1 and eta-min <= eta-max")
        values = [low] if args.eta_points == 1 else [low + (high-low)*k/(args.eta_points-1) for k in range(args.eta_points)]
    if not values or any(not v.is_finite() or v < 0 or v > 1 for v in values):
        raise ValueError("All eta values must be finite and in [0,1]")
    return [str(v.normalize()) for v in sorted(set(values))]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.min_sensors <= args.max_sensors <= 5:
        raise ValueError("This exhaustive implementation supports 1 <= min <= max <= 5")
    if not all(np.isfinite(x) and x > 0 for x in (args.eig_rtol, args.range_tol, args.tie_tol)):
        raise ValueError("Numerical tolerances must be finite and positive")
    if args.log_every < 0:
        raise ValueError("log-every must be nonnegative")
    eta_strings = eta_grid(args)
    etas = np.array([float(v) for v in eta_strings])
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Nonempty output: {output}; choose --output or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    start, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    files = sorted(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    # Keep public provenance portable: omit host-specific absolute paths.
    portable_output = str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else output.name
    config = {**vars(args), "output": portable_output, "eta_grid": eta_strings,
              "fixed_before_loss": True, "ancillas": 0, "target": "arithmetic_mean",
              "phase_generator": "Z/2", "loss": "independent_flagged_erasure",
              "phase_location": "after_loss", "criterion": "inverse_mean_QCRB_per_generated_photon"}
    all_rows, class_rows, statistics, all_winners = [], [], [], {}
    for n in range(args.min_sensors, args.max_sensors + 1):
        catalogue = search_catalogue(n, args.graph_reduction)
        probabilities = loss_weights(n, etas)
        best = np.full(len(etas), -np.inf)
        witnesses = []
        seen, best_profiles, candidates = set(), {}, 0
        for item in catalogue:
            graph_mask = item["graph_mask"]
            class_best = np.zeros(len(etas))
            # Global profile deduplication preserves its FIRST physical witness.
            # Per-class records are computed even for globally repeated profiles.
            for axes, profile in signed_profiles(n, item["edges"]):
                candidates += 1
                key = profile.tobytes()
                if key not in best_profiles:
                    q = np.einsum("es,sij->eij", probabilities, profile)
                    scores = effective_information(q, args.eig_rtol, args.range_tol) / n
                    best_profiles[key] = scores
                else:
                    scores = best_profiles[key]
                class_best = np.maximum(class_best, scores)
                if key not in seen:
                    seen.add(key)
                    witnesses.append((graph_mask, axes, profile.copy(), scores))
                    best = np.maximum(best, scores)
                if args.log_every and candidates % args.log_every == 0:
                    print(json.dumps({"stage": "search", "n": n, "candidates": candidates,
                                      "unique_profiles": len(seen)}, ensure_ascii=False), flush=True)
            for e, eta in enumerate(eta_strings):
                class_rows.append({"n": n, "graph_mask": graph_mask, "eta": eta,
                                   "best_information_per_photon": float(class_best[e])})
        expected = len(catalogue) * 6 ** n
        if candidates != expected:
            raise ArithmeticError("Candidate count mismatch")
        statistics.append({"n": n, "labeled_graphs": 1 << (n*(n-1)//2),
                           "graph_representatives": len(catalogue), "evaluated_candidates": candidates,
                           "expected_candidates": expected, "unique_qfi_profiles": len(seen),
                           "classes": list(catalogue)})
        for e, eta in enumerate(eta_strings):
            # Resolve near ties with exact rational arithmetic. No axis/sign
            # pattern is discarded on the basis of heuristic block structure.
            contenders = [w for w in witnesses if w[3][e] >= best[e]-args.tie_tol*max(1., best[e])]
            exact, chosen, tied = Fraction(-1), None, 0
            for witness in contenders:
                value, _ = exact_information(witness[2], eta)
                if value > exact:
                    exact, chosen, tied = value, witness, 1
                elif value == exact:
                    tied += 1
            graph_mask, axes, profile, scores = chosen
            numeric = scores[e] * n
            if not np.isclose(float(exact), numeric, rtol=1e-7, atol=1e-10):
                raise ArithmeticError("Rational winner check failed; examine rank tolerances")
            protocol_id = f"N{n}_g{graph_mask}_" + "_".join(a.replace("+", "p").replace("-", "m") for a in axes)
            all_winners[protocol_id] = {**describe(n, graph_mask, axes),
                                       "qfi_by_survivor_mask": profile.tolist()}
            score = exact / n
            eta_fraction = Fraction(eta)
            all_rows.append({"n": n, "eta": eta, "protocol_id": protocol_id,
                             "information_per_photon": float(score), "information_per_photon_exact": str(score),
                             "effective_information": float(exact), "effective_information_exact": str(exact),
                             "mean_QCRB_per_trial": float(1/exact) if exact else "inf",
                             "independent_per_photon": float(eta_fraction),
                             "ghz_per_photon": float(n * eta_fraction ** n),
                             "gain_over_independent": float(score/eta_fraction) if eta_fraction else "",
                             "tied_unique_profiles_exact_among_contenders": tied})
        print(json.dumps({"stage": "completed_n", "n": n, "candidates": candidates,
                          "unique_profiles": len(seen)}, ensure_ascii=False), flush=True)
    summary = {"configuration": config, "search_statistics": statistics,
               "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter()-start,
               "environment": {"python": platform.python_version(), "numpy": np.__version__},
               "source_sha256": hashes, "argv": sys.argv if argv is None else argv,
               "optimality": "exhaustive numerical optimum within configured tolerances, fixed local Clifford class only",
               "tie_policy": "numerical contenders compared exactly; first rational maximizer retained; ties count distinct labeled QFI profiles, not circuits"}
    write_csv(output / "best_by_n_eta.csv", all_rows)
    write_csv(output / "best_by_lc_class.csv", class_rows)
    (output / "protocols.json").write_text(json.dumps(all_winners, indent=2)+"\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"stage": "finished", "output": str(output), "seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
