"""Exhaustive fixed-Clifford graph search with an upper bound on total photons.

Every physical photon is independently erased with probability 1-eta. Its
node's phase is encoded after erasure. Input Clifford gates do not depend on
the survival record. Classical records of final survival positions are kept.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations, product
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from fixed_clifford_sensors.engine import AXES, GATES, unsigned_profiles
from fixed_clifford_sensors_N6_7.core import catalogue, aligned_signs, aggregate_profile


def scalar_information(q):
    """Inverse SLD CR bound for the arithmetic mean of the logical phases."""
    q = np.asarray(q, dtype=float)
    n = q.shape[-1]
    vals, vecs = np.linalg.eigh(q)
    scale = np.max(np.abs(vals), axis=-1, keepdims=True)
    if np.any(vals < -1e-10 * np.maximum(scale, 1e-20)):
        raise ArithmeticError("Negative QFIM eigenvalue")
    active = vals > 1e-11 * scale
    coords = np.sum(vecs, axis=-2) / n
    good = np.linalg.norm(np.where(active, 0, coords), axis=-1) < 1e-8
    cost = np.sum(np.divide(coords**2, vals, out=np.zeros_like(vals), where=active), axis=-1)
    return np.divide(1, cost, out=np.zeros_like(cost), where=good & (cost > 0))


def exact_information(q):
    n = len(q)
    w = [Fraction(1, n)] * n
    rows = [list(row) + [v] for row, v in zip(q, w)]
    pivots = []
    r = 0
    for col in range(n):
        pivot = next((i for i in range(r, n) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        divisor = rows[r][col]
        rows[r] = [v / divisor for v in rows[r]]
        for i in range(n):
            if i != r:
                a = rows[i][col]
                rows[i] = [x - a*y for x, y in zip(rows[i], rows[r])]
        pivots.append(col)
        r += 1
    if any(not any(row[:-1]) and row[-1] for row in rows):
        return Fraction(0)
    solution = [Fraction(0)] * n
    for i, p in enumerate(pivots):
        solution[p] = rows[i][-1]
    cost = sum(x*y for x, y in zip(w, solution))
    return 1 / cost if cost else Fraction(0)


def exact_matrix(a, eta):
    eta = Fraction(eta)
    l = len(a) - 1
    weights = [eta**k * (1-eta)**(l-k) for k in range(l+1)]
    return [[sum(int(a[k, i, j])*weights[k] for k in range(l+1))
             for j in range(a.shape[-1])] for i in range(a.shape[-1])]




def sign_patterns(l):
    # An overall reversal of every axis does not change any QFIM.
    return np.array([(1,) + rest for rest in product((1, -1), repeat=l-1)], dtype=np.int16)


def physical_signatures(l):
    """Exhaust all graph LC classes and unsigned axes before assigning nodes.

    A signature is the MULTISET of physical rank-one blocks and their exact
    erasure-activity polynomials. Signs and all node assignments are restored
    later. No old independent-phase sign-identifiability pruning is used.
    """
    start = time.perf_counter()
    records = {}
    graph_stats = []
    for gid, graph in enumerate(catalogue(l)):
        choices, profiles = unsigned_profiles(l, graph["edges"])
        count = 0
        for axes, raw in zip(choices, profiles):
            # An isolated effective X has zero QFI and is dominated by Z.
            # Isolated Y and Z have identical loss QFIMs. Fixing Z loses no max.
            if any(adj == 0 and int(axis) != 2 for adj, axis in zip(graph["adjacent"], axes)):
                continue
            signs, blocks = aligned_signs(raw[-1])
            if signs is None:
                raise ArithmeticError("Unexpected inactive nonisolated generator")
            aligned = raw * signs[None, :, None] * signs[None, None, :]
            aggregate = aggregate_profile(aligned)
            pieces = []
            rebuilt = np.zeros_like(aggregate)
            for block in blocks:
                coeff = tuple(map(int, aggregate[:, block[0], block[0]]))
                pieces.append((len(block), coeff, tuple(block)))
                for i in block:
                    for j in block:
                        rebuilt[:, i, j] = coeff
            if not np.array_equal(rebuilt, aggregate):
                raise ArithmeticError("Physical QFIM is not the stated block polynomial")
            pieces.sort()
            key = tuple((size, coeff) for size, coeff, _ in pieces)
            if key not in records:
                records[key] = {
                    "graph_id": gid, "edges": graph["edges"],
                    "unsigned_axes": list(map(int, axes)),
                    "aligning_signs": signs.astype(int).tolist(),
                    "blocks": [list(block) for _, _, block in pieces],
                    "coefficients": [list(coeff) for _, coeff, _ in pieces],
                }
            count += 1
        graph_stats.append({"graph_id": gid, "evaluated_unsigned_axes": count})
        print(json.dumps({"stage": "physical_graph", "photons": l, "graph": gid+1,
                          "graphs": len(catalogue(l)), "signatures": len(records)}), flush=True)
    return list(records.values()), {"photons": l, "LC_graphs": len(catalogue(l)),
                                  "physical_signatures": len(records),
                                  "graphs": graph_stats, "seconds": time.perf_counter()-start}


def canonical_node_key(a, n):
    return min(a[:, p, :][:, :, p].tobytes() for p in permutations(range(n)))






def set_partitions(items):
    if not items:
        yield ()
        return
    first = items[0]
    for blocks in set_partitions(items[1:]):
        yield ((first,),) + blocks
        for i in range(len(blocks)):
            yield blocks[:i] + ((first,) + blocks[i],) + blocks[i+1:]




def evaluate_arrays(arrays, etas, photons):
    k = np.arange(photons+1)
    weights = np.asarray(etas)[:, None]**k * (1-np.asarray(etas)[:, None])**(photons-k)
    scores = []
    for w in weights:
        matrices = np.einsum("k,ckij->cij", w, arrays, optimize=True)
        scores.append(scalar_information(matrices)/photons)
    return np.asarray(scores)


def choose_exact(arrays, scores, eta, photons):
    near = np.flatnonzero(scores >= scores.max()-1e-10*max(1.,scores.max()))
    values = [(exact_information(exact_matrix(arrays[i], eta))/photons, int(i)) for i in near]
    best = max(x[0] for x in values)
    winner = next(i for value, i in values if value == best)
    return best, winner, sum(value == best for value, _ in values)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(None)
def node_groupings(l, n):
    """All surjections photons -> nodes, modulo permutation of entire nodes.

    Unlabeled nonempty set partitions include EVERY positive allocation,
    including unequal counts. Never require equal group sizes.
    """
    return tuple(blocks for blocks in set_partitions(tuple(range(l))) if len(blocks) == n)


def grouped_candidates(n, l, signatures):
    groups, signs = node_groupings(l, n), sign_patterns(l)
    arrays, witnesses, seen = [], [], set()
    generated = unidentifiable = 0
    for sid, record in enumerate(signatures):
        sizes = [len(b) for b in record['blocks']]
        labels = np.repeat(np.arange(len(sizes)), sizes)
        coeff = np.array(record['coefficients'], dtype=np.int16)
        for gid, grouping in enumerate(groups):
            node = np.zeros(l, dtype=int)
            for i, group in enumerate(grouping):
                node[list(group)] = i
            counts = np.zeros((len(signs), len(sizes), n), dtype=np.int16)
            for b in range(len(sizes)):
                for i in range(n):
                    counts[:, b, i] = signs[:, (labels == b) & (node == i)].sum(axis=1)
            candidates = np.einsum('bk,sbi,sbj->skij', coeff, counts, counts, optimize=True)
            generated += len(candidates)
            good = scalar_information(candidates[:, -1]) > 0
            unidentifiable += int((~good).sum())
            for sign_id in np.flatnonzero(good):
                a = candidates[sign_id]
                key = (tuple(sorted(map(len, grouping))), canonical_node_key(a, n))
                if key not in seen:
                    seen.add(key)
                    arrays.append(a)
                    witnesses.append((sid, gid, int(sign_id)))
    return np.asarray(arrays, dtype=np.int16), np.asarray(witnesses, dtype=np.int32), {
        'node_set_partitions': len(groups),
        'allocations_modulo_node_permutation': sorted(set(tuple(sorted(map(len, g))) for g in groups)),
        'relative_sign_patterns': len(signs),
        'generated_grouped_signed_profiles': generated,
        'unidentifiable_without_loss': unidentifiable,
        'unique_node_QFIM_polynomials': len(arrays),
    }


def recover_protocol(n, l, signatures, witness, eta):
    sid, gid, sign_id = map(int, witness)
    record = signatures[sid]
    groups = node_groupings(l, n)[gid]
    signs = sign_patterns(l)[sign_id]
    canonical_to_old = sum(record['blocks'], [])
    old_axes = [None] * l
    for k, old in enumerate(canonical_to_old):
        sign = int(signs[k]) * record['aligning_signs'][old]
        old_axes[old] = ('+' if sign > 0 else '-') + AXES[record['unsigned_axes'][old]]
    order = [canonical_to_old[k] for group in groups for k in group]
    relabel = {old: new for new, old in enumerate(order)}
    axes = [old_axes[old] for old in order]
    return {'nodes': n, 'total_photons': l, 'eta': eta,
            'photons_per_node': list(map(len, groups)),
            'photon_to_node': [i for i, group in enumerate(groups) for _ in group],
            'edges': sorted([sorted([relabel[i], relabel[j]]) for i, j in record['edges']]),
            'effective_axes': axes, 'clifford_gates': [GATES[a] for a in axes],
            'witness_indices': list(map(int, witness))}


def ghz_candidates(n, l):
    """Independent complete GHZ enumeration for every allocation and all signs.

    Fix a contiguous map per integer allocation. Enumerate all labeled GHZ
    partitions and signs on it; photon labels cover all other placements.
    """
    from math import comb
    allocations = sorted(set(tuple(sorted(map(len, g))) for g in node_groupings(l, n)))
    signs = sign_patterns(l)
    arrays, records, seen = [], [], set()
    for allocation in allocations:
        node = np.repeat(np.arange(n), allocation)
        for blocks in set_partitions(tuple(range(l))):
            counts = np.zeros((len(signs), len(blocks), n), dtype=np.int16)
            coeff = np.zeros((len(blocks), l+1), dtype=np.int16)
            for b, block in enumerate(blocks):
                for i in range(n):
                    ids = [p for p in block if node[p] == i]
                    counts[:, b, i] = signs[:, ids].sum(axis=1)
                for k in range(len(block), l+1):
                    coeff[b, k] = comb(l-len(block), k-len(block))
            candidates = np.einsum('bk,sbi,sbj->skij', coeff, counts, counts, optimize=True)
            for si in np.flatnonzero(scalar_information(candidates[:, -1]) > 0):
                a = candidates[si]
                key = (allocation, canonical_node_key(a, n))
                if key in seen:
                    continue
                seen.add(key)
                arrays.append(a)
                records.append({'allocation': list(allocation), 'photon_to_node': node.tolist(),
                                'photon_blocks': [list(b) for b in blocks],
                                'block_sizes': list(map(len, blocks)),
                                'signed_node_counts': counts[si].tolist(),
                                'signs': signs[si].tolist()})
    return np.asarray(arrays, dtype=np.int16), records


def independent_information(allocation, eta):
    """Independent |+> photons, unweighted node mean, per generated photon."""
    return Fraction(len(allocation)**2, sum(allocation)) * Fraction(eta) / sum(
        Fraction(1, m) for m in allocation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', default='2,3')
    parser.add_argument('--max-photons', type=int, default=6)
    parser.add_argument('--eta-min', default='0.05')
    parser.add_argument('--eta-max', default='1')
    parser.add_argument('--eta-points', type=int, default=20)
    parser.add_argument('--etas', help='Explicit comma-separated decimal survival probabilities')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    nodes = sorted(set(map(int, args.nodes.split(','))))
    if not nodes or not 1 <= min(nodes) <= max(nodes) <= args.max_photons <= 6:
        raise ValueError('Require 1 <= N <= max-photons <= 6 (bundled graph catalogue)')
    if args.etas:
        grid = sorted(set(Decimal(s.strip()) for s in args.etas.split(',')))
    else:
        lo, hi = Decimal(args.eta_min), Decimal(args.eta_max)
        if args.eta_points < 1 or lo > hi:
            raise ValueError('Require eta-points >= 1 and eta-min <= eta-max')
        grid = [lo] if args.eta_points == 1 else [lo+(hi-lo)*i/(args.eta_points-1) for i in range(args.eta_points)]
    if not grid or any(not e.is_finite() or not 0 <= e <= 1 for e in grid):
        raise ValueError('Invalid survival probability')
    grid = [str(e.normalize()) for e in grid]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    physical, stats, rows, protocols, allocation_rows = {}, [], [], {}, []
    for n in nodes:
        for l in range(n, args.max_photons+1):
            begin = time.perf_counter()
            if l not in physical:
                physical[l] = physical_signatures(l)
                (out/f'physical_signatures_L{l}.json').write_text(json.dumps(physical[l][0], separators=(',', ':')))
            signatures, physical_stats = physical[l]
            arrays, witnesses, candidate_stats = grouped_candidates(n, l, signatures)
            scores = evaluate_arrays(arrays, list(map(float, grid)), l)
            ga, gr = ghz_candidates(n, l)
            gs = evaluate_arrays(ga, list(map(float, grid)), l)
            allocation_labels = [tuple(sorted(map(len, node_groupings(l, n)[int(w[1])]))) for w in witnesses]
            for allocation in candidate_stats['allocations_modulo_node_permutation']:
                allocation = tuple(allocation)
                ai = np.array([i for i, a in enumerate(allocation_labels) if a == allocation])
                bi = np.array([i for i, g in enumerate(gr) if tuple(g['allocation']) == allocation])
                for e, eta in enumerate(grid):
                    av, local, aties = choose_exact(arrays[ai], scores[e, ai], eta, l)
                    gv, glocal, _ = choose_exact(ga[bi], gs[e, bi], eta, l)
                    if av != gv:
                        raise ArithmeticError(f'Graph/GHZ allocation difference at {(n, allocation, eta)}')
                    index, gi = int(ai[local]), int(bi[glocal])
                    key = f'N{n}_A' + '_'.join(map(str, allocation)) + f'_eta{eta}'
                    p = recover_protocol(n, l, signatures, witnesses[index], eta)
                    p.update({'R_exact': str(av), 'R': float(av),
                              'node_QFIM': [[str(x) for x in row] for row in exact_matrix(arrays[index], eta)],
                              'A_by_survivor_count': arrays[index].tolist(), 'GHZ_partition_witness': gr[gi]})
                    protocols[key] = p
                    allocation_rows.append({'N': n, 'L': l, 'allocation': '+'.join(map(str, allocation)),
                                            'eta': eta, 'R': float(av), 'R_exact': str(av),
                                            'effective_QFI': float(l*av), 'effective_QFI_exact': str(l*av),
                                            'best_GHZ_partition': '+'.join(map(str, sorted(gr[gi]['block_sizes']))),
                                            'independent_R': float(independent_information(allocation, eta)),
                                            'protocol': key})
            for e, eta in enumerate(grid):
                value, winner, ties = choose_exact(arrays, scores[e], eta, l)
                gv, gi, _ = choose_exact(ga, gs[e], eta, l)
                if value != gv:
                    raise ArithmeticError(f'Full graph/GHZ difference at {(n,l,eta)}: {value}, {gv}')
                protocol = recover_protocol(n, l, signatures, witnesses[winner], eta)
                key = f'N{n}_L{l}_eta{eta}'
                ghz = gr[gi]
                allocations = candidate_stats['allocations_modulo_node_permutation']
                independent = max(independent_information(a, eta) for a in allocations)
                protocol.update({'R_exact': str(value), 'R': float(value),
                                 'node_QFIM': [[str(x) for x in row] for row in exact_matrix(arrays[winner], eta)],
                                 'A_by_survivor_count': arrays[winner].tolist(),
                                 'GHZ_partition_witness': ghz})
                protocols[key] = protocol
                rows.append({'N': n, 'L': l, 'eta': eta, 'R': float(value), 'R_exact': str(value),
                             'effective_QFI': float(l*value), 'effective_QFI_exact': str(l*value),
                             'best_GHZ_partition': '+'.join(map(str, sorted(ghz['block_sizes']))),
                             'GHZ_allocation': '+'.join(map(str, ghz['allocation'])),
                             'graph_allocation': '+'.join(map(str, protocol['photons_per_node'])),
                             'best_independent_R_at_L': float(independent),
                             'gain_over_independent_at_L': float(value/independent) if independent else '',
                             'ties_in_unique_polynomials': ties, 'protocol': key})
            np.savez_compressed(out/f'candidates_N{n}_L{l}.npz', A=arrays, witnesses=witnesses,
                                ghz_A=ga, eta=np.array(list(map(float, grid))), scores=scores)
            stats.append({'N': n, 'L': l, **candidate_stats, 'ghz_unique_polynomials': len(ga),
                          'physical_search': physical_stats, 'seconds': time.perf_counter()-begin})
            print(json.dumps({'stage': 'case_done', **{k:v for k,v in stats[-1].items() if k != 'physical_search'}}), flush=True)
    global_rows, total_rows = [], []
    for n in nodes:
        for eta in grid:
            available = [r for r in rows if r['N'] == n and r['eta'] == eta]
            maximum = max(Fraction(r['R_exact']) for r in available)
            tied = [r for r in available if Fraction(r['R_exact']) == maximum]
            winner = min(tied, key=lambda r: r['L'])
            allocation_ties = [r['allocation'] for r in allocation_rows if r['N'] == n and r['eta'] == eta
                               and Fraction(r['R_exact']) == maximum]
            global_rows.append({**winner, 'max_photons': args.max_photons,
                                'all_tied_L': '+'.join(str(r['L']) for r in tied),
                                'all_tied_allocations': ';'.join(allocation_ties),
                                'independent_budget_R': float(Fraction(eta)),
                                'gain_over_independent_budget': float(maximum/Fraction(eta)) if Fraction(eta) else ''})
            best_total = max(Fraction(r['effective_QFI_exact']) for r in available)
            total_winner = min((r for r in available if Fraction(r['effective_QFI_exact']) == best_total), key=lambda r: r['L'])
            total_rows.append({**total_winner, 'max_photons': args.max_photons,
                               'selection_objective': 'maximum effective QFI per trial, no photon division'})
    write_csv(out/'best_by_N_L_eta.csv', rows)
    write_csv(out/'best_by_allocation_eta.csv', allocation_rows)
    write_csv(out/'best_under_budget.csv', global_rows)
    write_csv(out/'best_total_QFI_under_budget.csv', total_rows)
    # One complete protocol per line keeps this browsable in GitHub's viewer.
    (out/'protocols.json').write_text('{\n' + ',\n'.join(
        '  '+json.dumps(k)+': '+json.dumps(v, ensure_ascii=False, separators=(',', ':'))
        for k,v in protocols.items()) + '\n}\n')
    summary = {'configuration': {'nodes': nodes, 'max_total_generated_photons': args.max_photons,
                                'eta_grid': grid, 'edge_weight': 'pi', 'ancillas': 0,
                                'adaptive_input': False, 'setting_mixture': False,
                                'allocation': 'all positive allocations; zero at any unknown node cannot identify the mean',
                                'loss': 'independent polarization-blind flagged erasure with common eta',
                                'target': 'unweighted arithmetic mean of N independent unknown node phases',
                                'resource': 'actual L generated sensor photons, including lost ones',
                                'objective': 'R=1/(L*w^T*Qbar^+*w); w=(1/N,...,1/N); range checked',
                                'readout': 'SLD QFIM bound, without a local-measurement restriction',
                                'tie_policy': 'minimum L among exact best per-photon scores'},
               'search_statistics': stats, 'total_seconds': time.perf_counter()-start,
               'finished_utc': datetime.now(timezone.utc).isoformat(),
               'environment': {'python': platform.python_version(), 'numpy': np.__version__},
               'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(ROOT.rglob('*.py'))},
               'optimality_scope': 'All ordinary graph states, fixed local Cliffords, and photon allocations at each N,L, on the specified eta grid. Exact GHZ domination certificates are generated by verify.py.'}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'stage': 'finished', 'seconds': summary['total_seconds'], 'rows': len(rows), 'budget_rows': len(global_rows)}), flush=True)


if __name__ == '__main__':
    main()




