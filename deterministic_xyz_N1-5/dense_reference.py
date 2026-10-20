"""Independent dense-state verification of deterministic sensing policies."""
from functools import lru_cache
from itertools import product
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
