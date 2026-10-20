"""Exact rational Bernstein certificate for the N=4 continuous upper envelope.

For every exhaustively enumerated graph and every loss pattern, d=ones gives
an upper bound on the optimized effective QFI. All resulting degree-4 loss
polynomials lie under max(independent, fully protected clique, bare clique).
The interval proof uses rational coefficients, not a transmission grid.
"""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import gzip,json


def bernstein(poly,a,b):
    degree=len(poly)-1
    power=[sum(poly[j]*comb(j,k)*a**(j-k)*(b-a)**k for j in range(k,degree+1)) for k in range(degree+1)]
    return [sum(power[j]*F(comb(k,j),comb(degree,j)) for j in range(k+1)) for k in range(degree+1)]


def prove(poly,a,b,depth=0):
    coeff=bernstein(poly,a,b)
    if min(coeff)>=0:return True,depth,1
    if depth>=16:return False,depth,1
    c=(a+b)/2
    left=prove(poly,a,c,depth+1);right=prove(poly,c,b,depth+1)
    return left[0] and right[0],max(left[1],right[1]),left[2]+right[2]


def main():
    out=Path(__file__).resolve().parent/'results'
    with gzip.open(out/'catalog_N4.json.gz','rt') as f:data=json.load(f)
    unique=set()
    for m,owners in data['owners'].items():
        m=int(m)
        for row in owners:
            profile=data['profiles'][row['profile_id']];totals=[0]*5
            for s,cs in enumerate(profile,1):totals[s.bit_count()]+=max(c.bit_count() for c in cs)
            poly=[F(0)]*5
            for k in range(1,5):
                for j in range(5-k):poly[k+j]+=F(totals[k]*comb(4-k,j)*(-1)**j,4+m)
            unique.add(tuple(poly))
    intervals=[(F(0),F(1,3),[F(0),F(1),F(0),F(0),F(0)]),
               (F(1,3),F(1,2),[F(0),F(1,2),F(3,2),F(0),F(0)]),
               (F(1,2),F(1),[F(0),F(0),F(3),F(-3),F(4)])]
    depth=0;leaves=0;failures=[]
    for p in unique:
        for a,b,base in intervals:
            ok,d,count=prove([x-y for x,y in zip(base,p)],a,b)
            depth=max(depth,d);leaves+=count
            if not ok:failures.append(dict(polynomial=list(map(str,p)),interval=[str(a),str(b)]))
    result=dict(exact_rational_certificate=not failures,unique_graph_upper_polynomials=len(unique),
                interval_checks=3*len(unique),max_subdivision_depth=depth,bernstein_subintervals=leaves,
                failures=failures,scope='N=4, all simple CZ graphs, X/Y central measurements, local Cliffords, all M>=0; ideal model',
                optimum='max(eta, eta/2+3*eta^2/2, 3*eta^2-3*eta^3+4*eta^4)')
    (out/'four_sensor_certificate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
