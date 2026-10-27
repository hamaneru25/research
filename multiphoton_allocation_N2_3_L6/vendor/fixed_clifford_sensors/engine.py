"""Exhaustive no-ancilla graph/Clifford sensing, with exact stabilizer QFIMs.

QFIM entries before loss-probability averaging are signed integers. Qubit 0
is the leftmost tensor axis in the independent dense implementation.
"""
from functools import lru_cache
from itertools import combinations, permutations, product
from fractions import Fraction
import numpy as np

AXES = ("X", "Y", "Z")
# Gates multiply right-to-left. Every entry obeys C^dagger Z C = signed axis.
GATES = {"+X": "H", "-X": "HZ", "+Y": "HSdg", "-Y": "HS",
         "+Z": "I", "-Z": "X"}


@lru_cache(None)
def edge_pairs(n):
    return tuple(combinations(range(n), 2))


def edges_from_mask(n, mask):
    return [list(e) for k, e in enumerate(edge_pairs(n)) if mask >> k & 1]


@lru_cache(None)
def relabel_actions(n):
    pairs = edge_pairs(n)
    index = {e: k for k, e in enumerate(pairs)}
    return tuple(tuple(index[tuple(sorted((p[i], p[j])))] for i, j in pairs)
                 for p in permutations(range(n)))


@lru_cache(None)
def canonical(n, mask):
    present = [k for k in range(len(edge_pairs(n))) if mask >> k & 1]
    return min(sum(1 << action[k] for k in present) for action in relabel_actions(n))


def local_complement(n, mask, vertex):
    index = {e: k for k, e in enumerate(edge_pairs(n))}
    neighbors = [j for j in range(n) if j != vertex
                 and mask >> index[tuple(sorted((vertex, j)))] & 1]
    for e in combinations(neighbors, 2):
        mask ^= 1 << index[e]
    return mask


@lru_cache(None)
def graph_catalogue(n):
    """All simple graphs, including disconnected graphs, modulo LC and labels."""
    unlabeled = {canonical(n, g) for g in range(1 << len(edge_pairs(n)))}
    remaining = set(unlabeled)
    classes = []
    while remaining:
        representative = min(remaining)
        remaining.remove(representative)
        orbit, todo = [representative], [representative]
        while todo:
            g = todo.pop()
            for v in range(n):
                h = canonical(n, local_complement(n, g, v))
                if h in remaining:
                    remaining.remove(h)
                    orbit.append(h)
                    todo.append(h)
        classes.append({"graph_mask": representative,
                        "edges": edges_from_mask(n, representative),
                        "unlabeled_members": sorted(orbit)})
    return tuple(classes)


def search_catalogue(n, reduction="lc"):
    if reduction == "lc":
        return graph_catalogue(n)
    if reduction == "isomorphism":
        graphs = sorted({canonical(n, g) for g in range(1 << len(edge_pairs(n)))})
    elif reduction == "labeled":
        graphs = range(1 << len(edge_pairs(n)))
    else:
        raise ValueError("Unknown graph reduction")
    return tuple({"graph_mask": g, "edges": edges_from_mask(n, g),
                  "unlabeled_members": [g]} for g in graphs)


def stabilizer_group(n, edges):
    """Return (x,z,sign) for signed Hermitian Pauli stabilizers.

    Intermediate representation is i**phase X**x Z**z; canonical Y=iXZ.
    Signs are retained, including negative two-site correlations.
    """
    adjacent = [0] * n
    for i, j in edges:
        adjacent[i] |= 1 << j
        adjacent[j] |= 1 << i
    group = [(0, 0, 0)]
    for i in range(n):
        x2, z2 = 1 << i, adjacent[i]
        group += [(x ^ x2, z ^ z2, (p + 2 * (z & x2).bit_count()) % 4)
                  for x, z, p in group]
    out = []
    for x, z, p in group:
        exponent = (p - (x & z).bit_count()) % 4
        if exponent not in (0, 2):
            raise ArithmeticError("Non-Hermitian stabilizer")
        out.append((x, z, 1 if exponent == 0 else -1))
    return tuple(out)


def unsigned_profiles(n, edges):
    """QFIM for every 3**n unsigned axis choice and all 2**n survivor sets.

    For a flat-spectrum stabilizer density operator, a local Pauli/2 has
    diagonal QFI 1 iff it anticommutes with a retained stabilizer. Cross QFI
    is the signed two-site Pauli expectation when both directions are active.
    """
    group = stabilizer_group(n, edges)
    selections = np.array(list(product(range(3), repeat=n)), dtype=np.int8)
    profiles = np.zeros((len(selections), 1 << n, n, n), dtype=np.int8)
    local = [[(1 << i, 0), (1 << i, 1 << i), (0, 1 << i)] for i in range(n)]
    for survivors in range(1, 1 << n):
        retained = [(x, z, sign) for x, z, sign in group if (x | z) & ~survivors == 0]
        expectations = {(x, z): sign for x, z, sign in retained}
        active = np.zeros((n, 3), dtype=np.int8)
        for i in range(n):
            if not survivors >> i & 1:
                continue
            for a, (px, pz) in enumerate(local[i]):
                active[i, a] = any(((px & z).bit_count() + (pz & x).bit_count()) % 2
                                   for x, z, _ in retained)
            profiles[:, survivors, i, i] = active[i, selections[:, i]]
        for i, j in edge_pairs(n):
            values = np.zeros((3, 3), dtype=np.int8)
            for a, (xi, zi) in enumerate(local[i]):
                for b, (xj, zj) in enumerate(local[j]):
                    values[a, b] = active[i, a] * active[j, b] * expectations.get((xi | xj, zi | zj), 0)
            corr = values[selections[:, i], selections[:, j]]
            profiles[:, survivors, i, j] = corr
            profiles[:, survivors, j, i] = corr
    return selections, profiles


def signed_profiles(n, edges):
    selections, profiles = unsigned_profiles(n, edges)
    for indices, profile in zip(selections, profiles):
        for signs_tuple in product((1, -1), repeat=n):
            signs = np.array(signs_tuple, dtype=np.int8)
            axes = tuple(("+" if s > 0 else "-") + AXES[a] for s, a in zip(signs, indices))
            yield axes, profile * signs[None, :, None] * signs[None, None, :]


def loss_weights(n, etas):
    counts = np.array([s.bit_count() for s in range(1 << n)])
    etas = np.asarray(etas, dtype=float)
    return etas[:, None] ** counts * (1 - etas[:, None]) ** (n - counts)


def effective_information(matrices, eig_rtol=1e-10, range_tol=1e-8):
    """Inverse mean-phase QCRB; return zero if mean is not identifiable."""
    matrices = np.asarray(matrices, dtype=float)
    n = matrices.shape[-1]
    vals, vecs = np.linalg.eigh(matrices)
    scale = np.max(np.abs(vals), axis=-1, keepdims=True)
    if np.any(vals < -eig_rtol * scale):
        raise ArithmeticError("QFIM is not positive semidefinite")
    positive = vals > eig_rtol * scale
    coordinates = np.sum(vecs, axis=-2) / n
    identifiable = np.linalg.norm(np.where(positive, 0, coordinates), axis=-1) <= range_tol
    terms = np.divide(coordinates ** 2, vals, out=np.zeros_like(vals), where=positive)
    bounds = np.sum(terms, axis=-1)
    return np.divide(1., bounds, out=np.zeros_like(bounds), where=identifiable & (bounds > 0))


def exact_information(profile, eta_string):
    """Rational calculation for a selected profile, including singular range test."""
    profile = np.asarray(profile)
    n = profile.shape[-1]
    eta = Fraction(eta_string)
    q = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for s, matrix in enumerate(profile):
        probability = eta ** s.bit_count() * (1 - eta) ** (n - s.bit_count())
        for i in range(n):
            for j in range(n):
                q[i][j] += probability * int(matrix[i, j])
    w = [Fraction(1, n)] * n
    augmented = [row[:] + [target] for row, target in zip(q, w)]
    pivots, row = [], 0
    for col in range(n):
        pivot = next((k for k in range(row, n) if augmented[k][col]), None)
        if pivot is None:
            continue
        augmented[row], augmented[pivot] = augmented[pivot], augmented[row]
        divisor = augmented[row][col]
        augmented[row] = [x / divisor for x in augmented[row]]
        for k in range(n):
            if k != row:
                multiplier = augmented[k][col]
                augmented[k] = [a - multiplier * b for a, b in zip(augmented[k], augmented[row])]
        pivots.append(col)
        row += 1
    if any(all(x == 0 for x in r[:-1]) and r[-1] for r in augmented):
        return Fraction(0), q
    solution = [Fraction(0)] * n
    for k, col in enumerate(pivots):
        solution[col] = augmented[k][-1]
    bound = sum(a * b for a, b in zip(w, solution))
    return (1 / bound if bound else Fraction(0)), q


def describe(n, graph_mask, axes):
    return {"n": n, "graph_mask": graph_mask, "edges": edges_from_mask(n, graph_mask),
            "effective_axes": list(axes), "sensor_gates": [GATES[a] for a in axes]}
