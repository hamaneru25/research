"""Check QFIM reduction, positivity, loss limits and the per-trial upper bound."""
from pathlib import Path
import json
import numpy as np
from compare_fixed_graphs import information, matrix


def main():
    records=[]
    for n in range(4,11):
        for m in range(n+1):
            for eta in (1e-7,.001,.01,.1,.2,.4,.6,.8,.99,1.):
                f=matrix(n,m,eta);w=np.ones(n)/n
                assert np.linalg.eigvalsh(f).min()>-1e-9
                actual=1/(w@np.linalg.pinv(f,hermitian=True,rcond=1e-13)@w)
                expected=information(n,m,eta)['feff']
                error=abs(actual-expected)/expected
                assert error<1e-7,(n,m,eta,actual,expected)
                assert expected<=n*eta+n*(n-1)*eta*eta+1e-9
                records.append(error)
            assert abs(information(n,m,1)['feff']-n*n)<1e-9
            assert information(n,m,0)['feff']==0
        for eta in (.001,.02,.3,.6,.95):
            assert abs(information(n,n,eta)['feff']-(n*eta+n*(n-1)*eta*eta))<1e-10
    result=dict(cases=len(records),max_relative_error=max(records),
                positive_semidefinite=True,upper_bound_respected=True,
                eta_one_limit_verified=True,all_protected_limit_verified=True)
    out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
    (out/'algebra_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
