"""Independent dense mixed-state checks and exact optimality certificates."""
import argparse
import csv
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from math import comb
from pathlib import Path
import time

import numpy as np
from run_search import (ROOT, AXES, unsigned_profiles, exact_matrix, exact_information,
                        scalar_information, sign_patterns, ghz_candidates, evaluate_arrays, node_groupings)

I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]],dtype=complex)
Y = np.array([[0,-1j],[1j,0]],dtype=complex)
Z = np.diag([1,-1]).astype(complex)
H = np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
S = np.diag([1,1j])
GATES = {"I": I, "X":X, "H":H, "HZ":H@Z, "HSdg":H@S.conj().T, "HS":H@S}


def tensor(ops):
    result = np.ones((1,1),complex)
    for op in ops:
        result = np.kron(result,op)
    return result


def physical_state(l, edges, gates):
    bits = (np.arange(1<<l)[:,None] >> np.arange(l-1,-1,-1)) & 1
    parity = np.zeros(1<<l,dtype=int)
    for i,j in edges:
        parity ^= bits[:,i] * bits[:,j]
    psi = (1-2*parity).astype(complex)/np.sqrt(1<<l)
    return tensor([GATES[g] for g in gates])@psi


def dense_node_qfim(psi, n, photon_to_node, mask, phases=None):
    """Partial trace, physical Z rotations, density derivatives, then SLD."""
    l = len(photon_to_node)
    live = [i for i in range(l) if mask>>i&1]
    lost = [i for i in range(l) if not mask>>i&1]
    if not live:
        return np.zeros((n,n))
    amplitude = psi.reshape([2]*l).transpose(live+lost).reshape(1<<len(live),-1)
    rho = amplitude@amplitude.conj().T
    bits = (np.arange(1<<len(live))[:,None] >> np.arange(len(live)-1,-1,-1)) & 1
    z = 1-2*bits
    generators = np.stack([z[:, [j for j,i in enumerate(live) if photon_to_node[i]==node]].sum(axis=1)/2
                           for node in range(n)])
    if phases is not None:
        diagonal = np.exp(-1j*np.asarray(phases)@generators)
        rho = diagonal[:,None]*rho*diagonal.conj()[None,:]
    derivatives = [-1j*(g[:,None]-g[None,:])*rho for g in generators]
    vals,vectors = np.linalg.eigh(rho)
    vals = np.maximum(vals,0)
    den = vals[:,None]+vals[None,:]
    factors = np.divide(2.,den,out=np.zeros_like(den),where=den>1e-13)
    transformed = [vectors.conj().T@d@vectors for d in derivatives]
    return np.array([[np.sum(factors*(a*b.conj()).real) for b in transformed] for a in transformed])


@lru_cache(None)
def stabilizer_profile(l, edges, axes):
    choices, profiles = unsigned_profiles(l,list(edges))
    ix = 0
    for a in axes:
        ix = 3*ix + AXES.index(a[1])
    signs = np.array([1 if a[0]=="+" else -1 for a in axes])
    return profiles[ix]*signs[None,:,None]*signs[None,None,:]


def grouped_profile(n, photon_to_node, edges, axes):
    l = len(photon_to_node)
    p = np.eye(n, dtype=int)[photon_to_node]
    physical = stabilizer_profile(l, tuple(map(tuple, edges)), tuple(axes))
    return np.einsum("iu,sij,jv->suv", p, physical, p, optimize=True)


def labeled_small_crosscheck(n, allocation, etas):
    """No LC, graph-automorphism, grouping, or signature reductions, L<=4."""
    l=sum(allocation)
    edges=list(__import__('itertools').combinations(range(l),2))
    counts=np.array([s.bit_count() for s in range(1<<l)])
    signs=sign_patterns(l)
    projection=np.eye(n,dtype=np.int16)[np.repeat(np.arange(n), allocation)]
    signed_projections=signs[:,:,None]*projection[None,:,:]
    best=np.zeros(len(etas))
    candidates=0
    for graph_mask in range(1<<len(edges)):
        present=[e for k,e in enumerate(edges) if graph_mask>>k&1]
        _,raw=unsigned_profiles(l,present)
        aggregate=np.stack([raw[:,counts==k].sum(axis=1,dtype=np.int16) for k in range(l+1)],axis=1)
        a=np.einsum("siu,akij,sjv->askuv",signed_projections,aggregate,signed_projections,optimize=True)
        a=a.reshape(-1,l+1,n,n)
        best=np.maximum(best,evaluate_arrays(a,etas,l).max(axis=1))
        candidates+=len(a)
    return best,candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    out = args.results
    summary = json.loads((out/'summary.json').read_text())
    protocols = json.loads((out/'protocols.json').read_text())
    rows = list(csv.DictReader((out/'best_by_N_L_eta.csv').open()))
    allocation_rows = list(csv.DictReader((out/'best_by_allocation_eta.csv').open()))
    budget = list(csv.DictReader((out/'best_under_budget.csv').open()))
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    coefficient_checks = 0
    for path in sorted(out.glob('physical_signatures_L*.json')):
        for record in json.loads(path.read_text()):
            l = len(record['unsigned_axes'])
            for block, coeff in zip(record['blocks'], record['coefficients']):
                r = len(block)
                for k, c in enumerate(coeff):
                    upper = comb(l-r, k-r) if k >= r else 0
                    if not 0 <= c <= upper:
                        raise ArithmeticError('Physical block exceeds GHZ coefficient bound')
                    coefficient_checks += 1
    certificates = []
    exact_evaluations = 0
    for stat in summary['search_statistics']:
        n, l = stat['N'], stat['L']
        # Re-enumeration rather than trusting a cached candidate file.
        ga, records = ghz_candidates(n, l)
        for row in (r for r in allocation_rows if int(r['N']) == n and int(r['L']) == l):
            allocation = tuple(map(int, row['allocation'].split('+')))
            candidate_ids = [i for i, r in enumerate(records) if tuple(r['allocation']) == allocation]
            upper = max(exact_information(exact_matrix(ga[i], row['eta']))/l for i in candidate_ids)
            exact_evaluations += len(candidate_ids)
            if upper != Fraction(row['R_exact']):
                raise ArithmeticError('Complete exact GHZ upper bound differs from graph achievement')
            certificates.append({'N': n, 'L': l, 'allocation': row['allocation'], 'eta': row['eta'],
                                 'upper_R_exact': str(upper), 'achieved_R_exact': row['R_exact']})
    for row in rows:
        values = [Fraction(r['R_exact']) for r in allocation_rows if r['N'] == row['N'] and r['L'] == row['L'] and r['eta'] == row['eta']]
        if max(values) != Fraction(row['R_exact']):
            raise ArithmeticError('Selection over allocations failed')
    for row in budget:
        n, eta = int(row['N']), row['eta']
        selected = [r for r in rows if int(r['N']) == n and r['eta'] == eta]
        upper = max(Fraction(r['R_exact']) for r in selected)
        if upper != Fraction(row['R_exact']):
            raise ArithmeticError('Budget selection failed')
        tied = sorted(int(r['L']) for r in selected if Fraction(r['R_exact']) == upper)
        if tied != sorted(map(int, row['all_tied_L'].split('+'))) or min(tied) != int(row['L']):
            raise ArithmeticError('Tie/minimum-photon policy failed')
        tied_allocations = {r['allocation'] for r in allocation_rows if int(r['N']) == n and r['eta'] == eta and Fraction(r['R_exact']) == upper}
        if tied_allocations != set(row['all_tied_allocations'].split(';')):
            raise ArithmeticError('All tied allocations differ')
    total_rows = list(csv.DictReader((out/'best_total_QFI_under_budget.csv').open()))
    for row in total_rows:
        maximum = max(Fraction(r['effective_QFI_exact']) for r in rows if r['N'] == row['N'] and r['eta'] == row['eta'])
        if maximum != Fraction(row['effective_QFI_exact']):
            raise ArithmeticError('Total per-trial QFI selection failed')
    def identity(p):
        return (p['nodes'], tuple(p['photon_to_node']), tuple(map(tuple, p['edges'])), tuple(p['clifford_gates']))
    unique = {identity(p): p for p in protocols.values()}
    dense_profiles = {}
    branch_error = aggregate_error = score_error = 0.
    mask_count = 0
    for ident, p in unique.items():
        n, l = p['nodes'], p['total_photons']
        node_map = p['photon_to_node']
        if [node_map.count(i) for i in range(n)] != p['photons_per_node']:
            raise ArithmeticError('Allocation and physical node map differ')
        psi = physical_state(l, p['edges'], p['clifford_gates'])
        expected = grouped_profile(n, node_map, p['edges'], p['effective_axes'])
        dense = np.array([dense_node_qfim(psi, n, node_map, mask) for mask in range(1<<l)])
        mask_count += len(dense)
        branch_error = max(branch_error, float(np.max(np.abs(dense-expected))))
        aggregate = np.stack([dense[[mask.bit_count() == k for mask in range(1<<l)]].sum(axis=0) for k in range(l+1)])
        aggregate_error = max(aggregate_error, float(np.max(np.abs(aggregate-np.asarray(p['A_by_survivor_count'])))))
        dense_profiles[ident] = dense
    for row in rows + allocation_rows:
        p = protocols[row['protocol']]
        l = p['total_photons']; eta = float(row['eta'])
        weights = np.array([eta**s.bit_count()*(1-eta)**(l-s.bit_count()) for s in range(1<<l)])
        q = np.einsum('s,sij->ij', weights, dense_profiles[identity(p)])
        score_error = max(score_error, abs(float(scalar_information(q)/l)-float(row['R'])))
    random_error = 0.; random_checks = 0
    for stat in summary['search_statistics']:
        n, l = stat['N'], stat['L']
        groups = node_groupings(l, n)
        for _ in range(10):
            group = groups[int(rng.integers(len(groups)))]
            node_map = [next(i for i, g in enumerate(group) if photon in g) for photon in range(l)]
            edges = [(i,j) for i in range(l) for j in range(i+1,l) if rng.random() < .45]
            axes = list(rng.choice(['+X','-X','+Y','-Y','+Z','-Z'], size=l))
            from run_search import GATES as GATE_NAMES
            psi = physical_state(l, edges, [GATE_NAMES[a] for a in axes])
            expected = grouped_profile(n, node_map, edges, axes)
            phases = rng.uniform(-1.5, 1.5, n)
            masks = sorted(set([0, (1<<l)-1, int(rng.integers(1<<l)), int(rng.integers(1<<l))]))
            for mask in masks:
                q = dense_node_qfim(psi, n, node_map, mask, phases)
                random_error = max(random_error, float(np.max(np.abs(q-expected[mask]))))
                random_checks += 1
    unreduced = []
    for stat in summary['search_statistics']:
        n, l = stat['N'], stat['L']
        if l > 4:
            continue
        selected = [r for r in rows if int(r['N']) == n and int(r['L']) == l]
        etas = [float(r['eta']) for r in selected]
        best = np.zeros(len(etas)); count = 0
        for allocation in stat['allocations_modulo_node_permutation']:
            value, number = labeled_small_crosscheck(n, allocation, etas)
            best = np.maximum(best, value); count += number
            allocation_name = '+'.join(map(str, allocation))
            allocation_expected = {r['eta']: float(r['R']) for r in allocation_rows if int(r['N']) == n and r['allocation'] == allocation_name}
            if np.max(np.abs(value-np.array([allocation_expected[r['eta']] for r in selected]))) > 1e-9:
                raise ArithmeticError('Unreduced search differs for a specific allocation')
        error = float(np.max(np.abs(best-np.array([float(r['R']) for r in selected]))))
        if error > 1e-9:
            raise ArithmeticError('Unreduced full graph/signed-axis/allocation search differs')
        unreduced.append({'N': n, 'L': l, 'allocations': stat['allocations_modulo_node_permutation'],
                          'labeled_signed_axis_candidates': count, 'max_R_error': error})
    if max(branch_error, aggregate_error, score_error, random_error) > 1e-9:
        raise ArithmeticError('Dense mixed-state SLD verification failed')
    simple_stats = None
    simple_path = out/'simple_GHZ_graphs.json'
    if simple_path.exists():
        simple = json.loads(simple_path.read_text())
        simple_cache = {}
        simple_error = 0.; simple_masks = 0
        for p in simple.values():
            ident = identity(p); l = p['total_photons']; n = p['nodes']
            if ident not in simple_cache:
                psi = physical_state(l, p['edges'], p['clifford_gates'])
                dense = np.array([dense_node_qfim(psi, n, p['photon_to_node'], mask) for mask in range(1<<l)])
                simple_cache[ident] = dense; simple_masks += len(dense)
            eta = float(p['eta'])
            weights = np.array([eta**s.bit_count()*(1-eta)**(l-s.bit_count()) for s in range(1<<l)])
            value = scalar_information(np.einsum('s,sij->ij', weights, simple_cache[ident]))/l
            simple_error = max(simple_error, abs(float(value)-p['R']))
        if simple_error > 1e-9:
            raise ArithmeticError('Simple GHZ physical circuit verification failed')
        simple_stats = {'unique_inputs': len(simple_cache), 'dense_erasure_masks': simple_masks,
                        'saved_values': len(simple), 'max_R_error': simple_error}
    result = {'all_passed': True, 'seed': args.seed,
              'exact_physical_block_coefficient_checks': coefficient_checks,
              'all_GHZ_candidates_exact_evaluations': exact_evaluations,
              'unique_winner_inputs': len(unique), 'dense_all_erasure_masks_checked': mask_count,
              'saved_N_L_eta_values_checked': len(rows), 'budget_values_checked': len(budget),
              'allocation_eta_values_checked': len(allocation_rows), 'total_QFI_budget_values_checked': len(total_rows),
              'max_dense_branch_QFIM_error': branch_error,
              'max_dense_aggregate_error': aggregate_error, 'max_saved_R_error': score_error,
              'random_nonzero_phase_mask_checks': random_checks,
              'max_random_nonzero_phase_QFIM_error': random_error,
              'unreduced_search_crosschecks': unreduced,
              'simple_GHZ_physical_circuits': simple_stats,
              'grid_upper_bound_certificates': certificates,
              'certificate_scope': 'All ordinary CZ graphs, fixed local Cliffords, all photon allocations, and all L between N and max_photons, on the requested eta grid. Physical block domination plus complete exact signed-GHZ enumeration.',
              'seconds': time.perf_counter()-start}
    (out/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'grid_upper_bound_certificates'}), flush=True)


if __name__ == '__main__':
    main()

