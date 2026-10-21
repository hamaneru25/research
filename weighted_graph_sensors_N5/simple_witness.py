"""Six-edge weighted K2,3 witness beating the ordinary-graph optimum."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import numpy as np
from engine import qfi_tensor,fixed_clifford_qfim,loss_weights,effective_information
from verify import dense_profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eta',type=float,default=.5)
    p.add_argument('--phi',type=float,default=2/3,help='Positive edge angle in radians; other edges have -phi')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not np.isfinite(a.eta) or not 0<a.eta<=1 or not np.isfinite(a.phi):raise ValueError('Invalid eta or phi')
    phi=a.phi
    weights=np.array([phi,phi,0,phi,0,-phi,0,-phi,0,-phi])
    axes=[2,1,1,2,1];signs=[1,-1,-1,-1,-1]
    T=qfi_tensor(5,weights);Q=fixed_clifford_qfim(T,5,axes,signs)
    avg=np.einsum('s,sij->ij',loss_weights(5,[a.eta])[0],Q)
    dense=dense_profile(5,weights,axes,signs)
    error=float(np.max(np.abs(dense-Q)))
    if error>1e-10:raise ArithmeticError('Independent physical mixed-state validation failed')
    R=float(effective_information(avg)/5)
    result={'eta':a.eta,'phi_rad':phi,'n':5,'positive_edges':[[0,1],[0,2],[0,4]],
            'negative_edges':[[1,3],[2,3],[3,4]],'other_edges':0,
            'weights_rad':weights.tolist(),'effective_axes':['+Z','-Y','-Y','-Z','-Y'],
            'sensor_gates':['I','HS','HS','X','HS'],'gate_order':'right to left',
            'R':R,'mean_QCRB_per_trial':1/(5*R),'mean_QFIM':avg.tolist(),
            'maximum_independent_dense_QFIM_error':error}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
