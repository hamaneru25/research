"""Graph-sensing foundation: heralded loss, central Pauli measurement, QFIM.

Sensors are vertices 0..N-1; retained central qubits are N..N+M-1.
Loss is BEFORE phase encoding. Loss locations and central outcomes are known.
Independent local phase gates are exp(-i phi_i n_i), n_i=(I-Z_i)/2.
This small-system dense reference implementation is not a global optimizer.
"""
from functools import lru_cache
from itertools import product
import time
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULI = {"X": X, "Y": Y, "Z": Z}


def tensor(ops):
    out = np.ones((1, 1), dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def graph_state(n, edges):
    """Normalized CZ graph-state vector; vertex 0 is the leftmost tensor axis."""
    if n < 1:
        raise ValueError("n must be positive")
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    parity = np.zeros(2**n, dtype=np.int8)
    seen = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("Invalid simple-graph edge")
        edge = tuple(sorted((u, v)))
        if edge in seen:
            raise ValueError("Duplicate edge")
        seen.add(edge)
        parity ^= (bits[:, u] * bits[:, v]).astype(np.int8)
    return ((-1.0)**parity / np.sqrt(2**n)).astype(complex)


def conditional_sensor_states(psi, n_sensors, n_central, survivors, bases):
    """Return (central signs, probability given loss, normalized sensor rho).

    Missing sensors are traced out, not replaced by a pure deleted graph.
    Central bases are chosen before measuring the central qubits in this trial.
    Survivor tensor order is increasing sensor label. Zero-probability outcomes
    are omitted. Central outcomes are retained as classical information.
    """
    if n_sensors < 1 or n_central < 0:
        raise ValueError("Invalid vertex counts")
    survivors = tuple(sorted(survivors))
    if len(set(survivors)) != len(survivors) or any(i not in range(n_sensors) for i in survivors):
        raise ValueError("Invalid survivors")
    if len(bases) != n_central or any(b not in PAULI for b in bases):
        raise ValueError("One X/Y/Z basis is required per central qubit")
    n = n_sensors + n_central
    psi = np.asarray(psi, complex).reshape(-1)
    if psi.size != 2**n or not np.isclose(np.vdot(psi, psi).real, 1):
        raise ValueError("psi must have the correct dimension and unit norm")
    lost = [i for i in range(n_sensors) if i not in survivors]
    order = list(range(n_sensors, n)) + list(survivors) + lost
    d, dl = 2**len(survivors), 2**len(lost)
    state = psi.reshape([2]*n).transpose(order).reshape(2**n_central, d*dl)
    eigenbras = {
        "X": np.array([[1, 1], [1, -1]], complex)/np.sqrt(2),
        "Y": np.array([[1, -1j], [1, 1j]], complex)/np.sqrt(2),
        "Z": I,
    }
    projected = (tensor([eigenbras[b] for b in bases]) @ state).reshape(2**n_central, d, dl)
    branches = []
    for outcome, amplitude in enumerate(projected):
        rho = amplitude @ amplitude.conj().T
        probability = float(np.trace(rho).real)
        if probability <= 1e-12:
            continue
        signs = tuple(1 if (outcome >> (n_central-1-j)) & 1 == 0 else -1 for j in range(n_central))
        branches.append((signs, probability, rho/probability))
    if not np.isclose(sum(p for _, p, _ in branches), 1):
        raise ArithmeticError("Central-outcome probabilities do not sum to one")
    return branches


@lru_cache(None)
def local_generators(k, axes):
    """Pauli/2 generators; using Z/2 instead of n changes only a common sign."""
    if len(axes) != k or any(a not in PAULI for a in axes):
        raise ValueError("Invalid local phase axes")
    return tuple(tensor([PAULI[axes[i]]/2 if j == i else I for j in range(k)]) for i in range(k))


def qufim(rho, generators):
    """SLD QFIM for a unitary family, valid for pure AND mixed states."""
    rho = np.asarray(rho, complex)
    if not np.isclose(np.trace(rho).real, 1) or not np.allclose(rho, rho.conj().T):
        raise ValueError("rho must be Hermitian and trace one")
    vals, vectors = np.linalg.eigh(rho)
    if vals.min() < -1e-9:
        raise ValueError("rho must be positive semidefinite")
    vals = np.clip(vals, 0, None)
    denom = vals[:, None] + vals[None, :]
    differences = vals[:, None] - vals[None, :]
    weights = np.divide(2*differences**2, denom, out=np.zeros_like(denom), where=denom > 1e-12)
    hs = np.array([vectors.conj().T @ g @ vectors for g in generators])
    if len(hs) == 0:
        return np.zeros((0, 0))
    weighted = hs.reshape(len(hs), -1) * np.sqrt(weights).reshape(1, -1)
    matrix = (weighted @ weighted.conj().T).real
    return (matrix + matrix.T)/2


def effective_information(matrix, weights=None):
    """Inverse QCRB for a weighted phase; zero when the target is unidentifiable.

    This is an SLD-QFI bound, not a claim of attainability by a fixed readout.
    """
    matrix = np.asarray(matrix, float)
    n = len(matrix)
    w = np.ones(n)/n if weights is None else np.asarray(weights, float)
    if matrix.shape != (n, n) or w.shape != (n,):
        raise ValueError("Invalid matrix or target weights")
    vals, U = np.linalg.eigh((matrix+matrix.T)/2)
    tol = 1e-10 * max(1.0, np.max(np.abs(vals)))
    if vals.min() < -tol:
        raise ValueError("Information matrix must be positive semidefinite")
    coordinates = U.T @ w
    positive = vals > tol
    if np.linalg.norm(coordinates[~positive]) > 1e-8:
        return 0.0
    variance_bound = np.sum(coordinates[positive]**2 / vals[positive])
    return float(1/variance_bound) if variance_bound > 0 else float("inf")


def loss_patterns(n_sensors, eta):
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie in [0,1]")
    for arrived in product((False, True), repeat=n_sensors):
        survivors = tuple(i for i, ok in enumerate(arrived) if ok)
        p = eta**len(survivors) * (1-eta)**(n_sensors-len(survivors))
        yield survivors, p


def branch_averaged_qufim(psi, n_sensors, n_central, survivors, bases, correction=None):
    """Conditional QFIM, including recorded central outcomes.

    Optional correction(survivors, bases, signs, rho) returns the physically
    corrected rho BEFORE phase encoding. No correction is silently assumed.
    """
    survivors = tuple(sorted(survivors))
    k = len(survivors)
    averaged = np.zeros((n_sensors, n_sensors))
    branches = conditional_sensor_states(psi, n_sensors, n_central, survivors, bases)
    if k == 0:
        return averaged
    for signs, probability, rho in branches:
        if correction is not None:
            rho = correction(survivors, tuple(bases), signs, rho)
        local = qufim(rho, local_generators(k, tuple("Z"*k)))
        averaged[np.ix_(survivors, survivors)] += probability*local
    return averaged


def build_qufim_cache(n_sensors, n_central, edges, allowed_bases=("X", "Y"), correction=None):
    """Build per-loss/per-basis QFIM data. Does not optimize measurement policy."""
    psi = graph_state(n_sensors+n_central, edges)
    cache = {}
    for survivors, _ in loss_patterns(n_sensors, .5):
        for bases in product(allowed_bases, repeat=n_central):
            cache[(survivors, bases)] = branch_averaged_qufim(psi, n_sensors, n_central, survivors, bases, correction)
    return cache


def evaluate_policy(cache, n_sensors, eta, policy):
    """Average QFIM FIRST, then apply the inverse-based target criterion."""
    total = np.zeros((n_sensors, n_sensors))
    for survivors, p in loss_patterns(n_sensors, eta):
        total += p*cache[(survivors, tuple(policy[survivors]))]
    return total, effective_information(total)


def validate_core():
    plus = np.array([1, 1], complex)/np.sqrt(2)
    assert np.allclose(qufim(np.outer(plus, plus.conj()), [Z/2]), [[1]])
    assert np.allclose(qufim(I/2, [Z/2]), [[0]])
    assert np.allclose(qufim(np.diag([.7, .3]), [X/2]), [[.16]])
    assert effective_information(np.diag([1., 0.])) == 0
    ghz = np.zeros(8, complex); ghz[0] = ghz[-1] = 1/np.sqrt(2)
    ghz_rho = np.outer(ghz, ghz.conj())
    assert np.allclose(qufim(ghz_rho, local_generators(3, ("Z",)*3)), np.ones((3,3)))
    assert np.isclose(effective_information(np.ones((3,3))), 9)
    psi = graph_state(4, [(3,0),(3,1),(3,2)])
    for signs, p, rho in conditional_sensor_states(psi, 3, 1, (0,1,2), ("X",)):
        gate = tensor([H,H,H])
        corrected = gate@rho@gate.conj().T
        if signs[0] == -1:
            gate = tensor([Z,I,I]); corrected = gate@corrected@gate.conj().T
        assert np.isclose(p, .5)
        assert np.allclose(corrected, ghz_rho)
    for _, _, rho in conditional_sensor_states(psi, 3, 1, (0,1), ("X",)):
        gate = tensor([H,H]); corrected = gate@rho@gate.conj().T
        assert np.allclose(qufim(corrected, local_generators(2, ("Z",)*2)), np.zeros((2,2)))
    print("検証OK：単一量子ビット、混合状態、GHZ、結果別補正、損失、識別可能性")


if __name__ == "__main__":
    validate_core()
    n_sensors = n_central = 3
    edges = [(3,0),(3,1),(3,4),(4,2),(4,5)]
    start = time.perf_counter()
    cache = build_qufim_cache(n_sensors, n_central, edges)
    policy = {survivors: ("X",)*n_central for survivors, _ in loss_patterns(n_sensors, .5)}
    Q, eff = evaluate_policy(cache, n_sensors, .7, policy)
    print("基礎データ数:", len(cache), "（1グラフ・補正なし）")
    print("例のQFIM（中央XXX固定、透過率0.7）:\n", np.round(Q,6))
    print("平均位相の有効QFI:", round(eff,6))
    print("経過時間:", round(time.perf_counter()-start,3), "秒")
