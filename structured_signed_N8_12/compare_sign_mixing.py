"""Controlled comparison of mixed and hub-uniform signs in seven saved winners.

Keep graph and absolute edge weight fixed. Remove the signs on the Y-side
vertices so that every Z-side hub has only one outgoing sign. Compare both
the corresponding Clifford axes and all axes uniform within the original
vertex groups. Independently recompute the selected protocols with SLD action
for every individual loss mask and every independent sensor phase.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np
from engine import Kernel, effective
from search import expand
from dense_engine import layout, pauli_actions, batches

ROOT = Path(__file__).resolve().parent
CASES = [(8, .05), (8, .50), (9, .30), (10, .35), (11, .50),
         (12, .50), (12, .70)]


def fixed_score(kernel, profile, eta, axes):
    counts = np.arange(kernel.n + 1)
    tensor = np.einsum('k,kij->ij', eta**counts * (1-eta)**(kernel.n-counts), profile)
    gather = np.array([3*i + 'XYZ'.index(a[-1]) for i, a in enumerate(axes)])
    signs = np.array([1 if a[0] == '+' else -1 for a in axes])
    q = tensor[np.ix_(gather, gather)] * signs[:, None] * signs[None, :]
    r = float(effective(q, np.array(kernel.sizes)/kernel.n)/kernel.n)
    return r, q


def independent_sld(n, weights, axes, eta):
    """Full N by N QFIM; no grouping of sensor phases or loss masks."""
    psi = np.exp(1j * layout(n)[1] @ weights) / np.sqrt(1 << n)
    axis = np.array(['XYZ'.index(a[-1]) for a in axes])
    sign = np.array([1 if a[0] == '+' else -1 for a in axes])
    avg = np.zeros((n, n))
    groups = list(layout(n)[2])
    groups.append(([tuple(range(n))], np.array([(1 << n)-1]),
                   np.arange(1 << n)[None, :, None], np.array([np.arange(3*n)])))
    for sets, masks, index, _ in batches(groups, chunk=16):
        kept = np.array(sets)
        k = len(sets[0])
        U, s, _ = np.linalg.svd(psi[index], full_matrices=False)
        vals = s*s
        perm, phase = pauli_actions(k)
        indices = 3*np.arange(k)[None, :] + axis[kept]
        PV = phase[indices]*sign[kept][:, :, None]/2
        GV = PV[:, :, :, None]*U[np.arange(len(sets))[:, None, None], perm[indices], :]
        B = U.conj().transpose(0, 2, 1)[:, None] @ GV
        den = vals[:, :, None] + vals[:, None, :]
        ratio = np.divide(vals[:, :, None]-vals[:, None, :], den,
                          out=np.zeros_like(den), where=den > 1e-13)
        LU = U[:, None] @ (2j*ratio[:, None]*B) - 2j*(GV-U[:, None] @ B)
        q = np.einsum('biar,bjar,br->bij', LU.conj(), LU, vals).real
        probability = eta**k*(1-eta)**(n-k)
        for b, live in enumerate(sets):
            avg[np.ix_(live, live)] += probability*q[b]
    return float(effective(avg, np.ones(n)/n)/n), avg


def flip_axis(a, sign):
    return a if sign == 1 else ('-' if a[0] == '+' else '+') + a[-1]


def check(record):
    started = time.monotonic()
    n, eta, winner = record['n'], record['eta'], record['best']
    assert winner['family'] == 1
    sizes = winner['sizes']
    sides = winner['spec']['sides']
    signs = winner['spec']['signs']
    axes_mixed = winner['group_axes']
    phi = winner['parameters'][0]
    # A complete bipartite graph has two interchangeable parts. Designate
    # the Z-axis part as hubs in both protocols to make the comparison explicit.
    zside = next(s for s in (0, 1) if all(a[-1] == 'Z' for a, t in zip(axes_mixed, sides) if t == s))
    assert all(a[-1] == 'Y' for a, s in zip(axes_mixed, sides) if s != zside)
    hub_count = sum(k for k, s in zip(sizes, sides) if s == zside)
    new_signs = [s if t == zside else 1 for s, t in zip(signs, sides)]
    axes_unmixed = [a if t == zside else flip_axis(a, s)
                    for a, s, t in zip(axes_mixed, signs, sides)]
    kernel = Kernel(sizes)
    wm = winner['group_weights']
    wu = [0. if sides[i] == sides[j] else phi*new_signs[i]*new_signs[j]
          for i, j in kernel.pairs]
    profiles = [kernel.profile(w) for w in (wm, wu)]
    rm, qm = fixed_score(kernel, profiles[0], eta, axes_mixed)
    ru, qu = fixed_score(kernel, profiles[1], eta, axes_unmixed)
    ru_axes_unchanged, _ = fixed_score(kernel, profiles[1], eta, axes_mixed)
    # Both sign choices get the same original group partition and full
    # group-uniform Clifford catalogue, not different search budgets.
    opt_m = kernel.evaluate(wm, [eta])[0]
    opt_u = kernel.evaluate(wu, [eta])[0]
    del profiles, kernel
    full_axes_m = [a for a, k in zip(axes_mixed, sizes) for _ in range(k)]
    full_axes_u = [a for a, k in zip(axes_unmixed, sizes) for _ in range(k)]
    full_wm, full_wu = expand(sizes, wm), expand(sizes, wu)
    rsm, qsm = independent_sld(n, full_wm, full_axes_m, eta)
    rsu, qsu = independent_sld(n, full_wu, full_axes_u, eta)
    # Check the product-unitary relation directly on state vectors as well.
    bits = (np.arange(1 << n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    vertex_sides = np.repeat(sides, sizes)
    vertex_signs = np.repeat(signs, sizes)
    flip_vertices = np.where((vertex_sides != zside) & (vertex_signs == -1))[0]
    mask = sum(1 << (n-1-int(i)) for i in flip_vertices)
    psi_m = np.exp(1j*layout(n)[1] @ full_wm)/np.sqrt(1 << n)
    psi_u = np.exp(1j*layout(n)[1] @ full_wu)/np.sqrt(1 << n)
    diagonal = np.exp(1j*phi*len(flip_vertices)*(bits[:, vertex_sides == zside] @ vertex_signs[vertex_sides == zside]))
    state_error = float(np.max(abs(psi_u-diagonal*psi_m[np.arange(1 << n)^mask])))
    leaf_sign_counts = [sum(k for k, s, t in zip(sizes, signs, sides) if t != zside and s == sg)
                        for sg in (1, -1)]
    result = dict(n=n, eta=eta, hubs=hub_count, leaves=n-hub_count, phi_rad=phi,
                  group_sizes=sizes, group_sides=sides, mixed_group_signs=signs,
                  unmixed_group_signs=new_signs, mixed_group_weights=wm,
                  unmixed_group_weights=wu, mixed_group_axes=axes_mixed,
                  unmixed_group_axes=axes_unmixed,
                  positive_hub_mixed_edge_counts=leaf_sign_counts,
                  negative_hub_mixed_edge_counts=leaf_sign_counts[::-1],
                  R_saved=winner['R'], R_mixed=rm, R_unmixed=ru,
                  abs_R_difference=abs(rm-ru), relative_difference=abs(rm-ru)/rm,
                  R_unmixed_axes_unchanged=ru_axes_unchanged,
                  R_mixed_best_group_axes=opt_m['R'], R_unmixed_best_group_axes=opt_u['R'],
                  mixed_best_group_axes=opt_m['group_axes'], unmixed_best_group_axes=opt_u['group_axes'],
                  group_optimum_difference=abs(opt_m['R']-opt_u['R']),
                  max_group_QFIM_difference=float(np.max(abs(qm-qu))),
                  R_mixed_independent_SLD=rsm, R_unmixed_independent_SLD=rsu,
                  independent_SLD_R_difference=abs(rsm-rsu),
                  max_full_QFIM_difference=float(np.max(abs(qsm-qsu))),
                  max_SLD_vs_group_R_error=max(abs(rm-rsm), abs(ru-rsu)),
                  state_vector_relation_max_error=state_error,
                  seconds=time.monotonic()-started)
    assert max(result['abs_R_difference'], result['group_optimum_difference'],
               result['max_group_QFIM_difference'], result['max_full_QFIM_difference'],
               result['max_SLD_vs_group_R_error'], state_error, abs(rm-winner['R'])) < 1e-9
    return result


def main():
    records = json.loads((ROOT/'results/best.json').read_text())
    rows = []
    for n, eta in CASES:
        record = next(r for r in records if r['n'] == n and abs(r['eta']-eta) < 1e-10)
        row = check(record)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('n', 'eta', 'R_mixed', 'R_unmixed',
              'abs_R_difference', 'group_optimum_difference', 'max_full_QFIM_difference', 'seconds')}), flush=True)
    report = {'scope': 'Seven selected family-1 winners. Same graph and common absolute weight; matched fixed axes and exhaustive axes uniform within the same saved vertex groups. Full independent-phase SLD-QFIM for all loss records.',
              'tolerance': 1e-9, 'passed': True, 'results': rows}
    path = ROOT/'results/sign_mixing_check.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    fields = ['n', 'eta', 'hubs', 'leaves', 'phi_rad', 'R_mixed', 'R_unmixed',
              'abs_R_difference', 'group_optimum_difference', 'R_unmixed_axes_unchanged',
              'max_full_QFIM_difference', 'max_SLD_vs_group_R_error']
    with (ROOT/'results/sign_mixing_check.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    print(json.dumps({'passed': True, 'count': len(rows), 'output': str(path)}), flush=True)


if __name__ == '__main__':
    main()
