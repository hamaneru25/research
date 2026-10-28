# Signed complete-bipartite weighted sensing, N=8 through 20

The study compares a restricted, explicitly defined family. It is not an
exhaustive search over all weighted graphs or all local Clifford placements.

## Family and model

Choose h hubs and m=N-h leaves. There are all h*m hub-leaf edges and no
intra-part edges. A hub has sign s_i=+1 or -1, and its edges all have phase
s_i*phi. The number q of negative hubs is varied. Permutations of hubs make
only q relevant. Overall complex conjugation permits 0<=phi<=pi, and sign
reversal permits 0<=q<=floor(h/2). Both choices of the hub part are examined.

Every node is a sensing photon; there are no ancillary or reference nodes.
Prepare |+>^N, apply CP(s_i*phi) on the edges, then fixed local Cliffords.
Independent polarization-blind erasure follows, with survival eta. Finally
each surviving sensor acquires its independent unknown theta_i under Z_i/2.
Loss positions are recorded, but the input never depends on the loss record.

The allowed input rotations are symmetry-preserving: one effective Pauli
axis X,Y,Z on positive hubs, its conjugate by X on negative hubs, and a
uniform signed Pauli on leaves. Thus the physical effective axis on a negative
hub is X, -Y, or -Z respectively. The search includes all 18 relative-sign
axis choices for each h,q. It excludes different axis choices within a group
and independent signs on individual leaves.

The original K_{2,N-2} probe is included explicitly: one +phi hub with +Z,
one -phi hub with -Z, and all leaves with -Y. Its one-parameter optimum is
saved separately from the broader K2-topology envelope.

For fixed inputs average the mixed-state SLD QFIM over every loss record:
Qbar=sum_S eta^|S| (1-eta)^(N-|S|) Q_S. For w=ones(N)/N use
R=1/(N*w^T*Qbar^+*w), provided w is in the range; otherwise R=0.
The denominator counts all N generated photons including those lost.
Independent probes give R=eta. No concrete readout CFI is optimized.

## Exact symmetry compression

Let |psi_+> be the all-positive CP(phi) graph. The signed state satisfies

|psi_s> = [product_negative_hubs X_i] tensor D(-q*phi)^tensor_m |psi_+>,
where D(alpha)=diag(1,exp(i*alpha)). This is a mathematical change of frame;
it does not introduce additional physical non-Clifford preparation gates.
Conjugating the chosen physical Pauli axes maps the hub axis to a uniform
axis. The leaf X axis maps to cos(q*phi)X+sin(q*phi)Y, its Y axis to
-sin(q*phi)X+cos(q*phi)Y, and Z stays Z.

The transformed state is permutation-symmetric within the h and m parts.
For k and l survivors in the parts, write their Dicke excitation indices a,b
and lost excitation indices c,d. The exact survivor/lost amplitude matrix is

A_(a,b),(c,d) = sqrt[C(k,a) C(l,b) C(h-k,c) C(m-l,d) / 2^N]
                 * exp[i*phi*(a+c)*(b+d)].

The survivor density operator is A A^dagger. Its Hilbert-space dimension is
(k+1)(l+1), rather than 2^(k+l). Thin SVD yields all nonzero eigenvalues.
Collective generators on each part preserve these symmetric spaces, so
their QFI does not omit couplings into non-symmetric zero eigenspaces.

For G_i=J_x,J_y,J_z on either part, compute the general mixed-state formula

Q_ij = 4 Re Tr(rho G_i G_j)
       -8 sum_ab lambda_a lambda_b/(lambda_a+lambda_b)
              Re[(G_i)_ab (G_j)_ba].

This includes the zero-eigenspace contribution via the first moment term.
It is not the pure-state covariance formula applied to mixed states.
Combine each (k,l) with multiplicity C(h,k)C(m,l), and then its loss
probability eta^(k+l)(1-eta)^(N-k-l). The six collective generators yield
a 6x6 tensor from which all allowed axes are evaluated.

After averaging, permutation symmetry decouples within-part phase differences
from the two uniform part phases. Let B be the 2x2 QFIM for the two uniform
part phases, and u=(h/N,m/N). Then the full independent-phase mean bound is
R=1/(N*u^T*B^+*u), with the corresponding range check. This is not an
assumption that all N unknown phases are equal.

## GHZ partition benchmark

For disjoint GHZ block sizes k_1,...,k_b summing to N, each intact block
supplies Q=eta^k J_k and an incomplete block supplies zero phase QFI.
For the same mean target and generated-photon count,

R_GHZ = N / sum_j eta^(-k_j).

All integer partitions are optimized exactly via dynamic programming:
cost[0]=0, cost[n]=min_(1<=k<=n)(eta^(-k)+cost[n-k]).
Singleton blocks include the independent-probe benchmark.

## Numerical search and reproducibility

Install Python 3.10+, NumPy, SciPy, Matplotlib. From this directory run:

    python search.py --sensors 8,9,10,11,12,13,14,15,16,17,18,19,20 \
      --grid-points 129 --peaks 4 --workers 6 --output results
    python verify.py
    python validate_results.py --results results
    python make_report.py --results results

Adjust --sensors, --etas, --grid-points, --peaks, --xtol, --workers and
--output. The default eta grid is 0.05,...,1.00. The final run used 129
uniform phi grid points on [0,pi], then bounded scalar refinement around
up to four best sampled interior maxima for every part size and eta.
The objective is the envelope over all allowed q/axis choices; endpoints
are included. This procedure is numerical, not a proof of a global maximum
in phi. For original K2 both axis choices and q remain fixed as above.
--resume reuses already-computed family files; use it only with unchanged
settings, or use a new output directory after changing parameters.

## Files

- results/best.csv, best.json: all 260 N/eta winners and GHZ comparisons.
- family_Nn_aa.json: every bipartition's best value, actual hub orientation,
  q, physical axes, optimal weight, 2x2 QFIM, coarse grid and optimizer logs.
- summary.json: ranges, gains and validation summary.
- validation.json and extended_validation.json: independent dense checks.
- information.png/svg: representative N information curves.
- topology_and_gain.png/svg: best topology and GHZ gain at all 260 points.
- signed_bipartite_report.html: standalone report with downloadable code/data.

Results are best found within this signed complete-bipartite and symmetric
fixed-Clifford family. They do not establish optimality over arbitrary
graphs, arbitrary edge weights, arbitrary local rotations, or arbitrary
quantum probes. Small graph comparisons and denser initial grids are checks,
not proofs of global optimality.
