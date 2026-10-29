import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import sys,json,numpy as np,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from engine import Kernel,effective
from compare_sign_mixing import fixed_score,independent_sld
from search import expand
out={'small_phi':[],'cases':[],'density_checks':[]}
for n,h in [(6,2),(8,2),(12,4)]:
 k=Kernel([h//2,h//2,n-h]);ax=['+Z','-Z','-Y']
 for eta in [.1,.5]:
  vals=[]
  for phi in [.001,.0005]:
   r,_=fixed_score(k,k.profile([0,0,phi,0,-phi,0]),eta,ax);vals.append((r-eta)/phi)
  slope=2*vals[1]-vals[0];pred=h*(n-h)*eta**2/n
  out['small_phi'].append(dict(n=n,h=h,eta=eta,predicted=pred,extrapolated=slope,error=abs(pred-slope)))
for n,h,phi,eta in [(8,2,.6095870505934571,.5),(12,4,.4067837408649881,.5)]:
 sizes=[h//2,h//2,n-h];weights=expand(sizes,[0,0,phi,0,-phi,0]);axes=['+Z']*(h//2)+['-Z']*(h//2)+['-Y']*(n-h)
 r,q=independent_sld(n,weights,axes,eta);rp,qp=independent_sld(n,weights,axes,1.)
 c=np.cos(phi/2);g=np.sin(phi/2)*c**(h-1);b=(1-np.cos(phi)**h)/2
 expected=np.block([[np.eye(h),np.full((h,n-h),g)],[np.full((n-h,h),g),(1-b)*np.eye(n-h)+b*np.ones((n-h,n-h))]])
 diagR=float(effective(np.diag(np.diag(q)),np.ones(n)/n)/n)
 nointact=float(effective(q-eta**n*qp,np.ones(n)/n)/n)
 out['cases'].append(dict(n=n,h=h,eta=eta,phi=phi,R=r,R_diagonal_only=diagR,R_intact_only=eta**n*rp,R_excluding_intact=nointact,pure_formula_error=float(abs(qp-expected).max()),hub_diag=float(np.diag(q)[:h].mean()),leaf_diag=float(np.diag(q)[h:].mean()),hub_leaf=float(q[:h,h:].mean()),leaf_leaf=float((q[h:,h:].sum()-np.trace(q[h:,h:]))/((n-h)*(n-h-1))),singleton_hub_QFI=c**(2*(n-h)),singleton_leaf_QFI=c**(2*h)))
 print(out['cases'][-1],flush=True)
# Validate exact reduced-state formula on several loss positions, including unequal lost +/- hubs.
n,h,phi=6,2,.6437994
sg=np.array([1,-1]);bits=(np.arange(1<<n)[:,None]>>np.arange(n-1,-1,-1))&1
A=bits[:,:h]@sg;B=bits[:,h:].sum(1);psi=np.exp(1j*phi*A*B)/np.sqrt(1<<n)
for live in [[0,2,3],[1,4],[2],[0],[0,1,2,3,4]]:
 lost=[i for i in range(n) if i not in live];k=len(live)
 arr=psi.reshape([2]*n).transpose(live+lost).reshape(1<<k,-1);rho=arr@arr.conj().T
 x=(np.arange(1<<k)[:,None]>>np.arange(k-1,-1,-1))&1
 aa=sum((sg[v]*x[:,i] for i,v in enumerate(live) if v<h),start=np.zeros(1<<k));bb=sum((x[:,i] for i,v in enumerate(live) if v>=h),start=np.zeros(1<<k))
 da=aa[:,None]-aa[None,:];db=bb[:,None]-bb[None,:]
 pred=np.exp(1j*phi*(aa[:,None]*bb[:,None]-aa[None,:]*bb[None,:]))/2**k
 for v in lost: pred*=(1+np.exp(1j*phi*(sg[v]*db if v<h else da)))/2
 out['density_checks'].append(dict(live=live,max_error=float(abs(rho-pred).max())))
path=ROOT/'results/bipartite_mechanism_checks.json';path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
out['edge_ablation']=[]
n,h,eta=6,2,.5
pairs=list(itertools.combinations(range(n),2));axes=['+Z','-Z']+['-Y']*4
for name in ['bipartite','one_cross_edge_removed','all_internal_edges_added']:
 slopes=[]
 for ph in [.001,.0005]:
  w=np.array([ph*(1 if i==0 else -1) if i<h<=j else (ph if name=='all_internal_edges_added' else 0.) for i,j in pairs])
  if name=='one_cross_edge_removed':w[pairs.index((0,2))]=0
  r,_=independent_sld(n,w,axes,eta);slopes.append((r-eta)/ph)
 count=7 if name=='one_cross_edge_removed' else 8
 out['edge_ablation'].append(dict(graph=name,predicted_slope=count*eta**2/n,extrapolated_slope=2*slopes[1]-slopes[0]))
path.write_text(json.dumps(out,indent=2)+'\n')
print(out['edge_ablation'],flush=True)

assert max(x['max_error'] for x in out['density_checks']) < 1e-12
assert max(x['pure_formula_error'] for x in out['cases']) < 1e-12
assert max(x['error']/x['predicted'] for x in out['small_phi']) < 1e-5
