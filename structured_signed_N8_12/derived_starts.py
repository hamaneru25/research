"""Additional N6/N7-inspired fully variable clique starts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from search import dense_job,ROOT

if __name__=='__main__':
    jobs=[(n,3,910000+n*100,2,str(ROOT/'results'))for n in range(8,13)]
    with ProcessPoolExecutor(max_workers=1)as pool:
        for f in as_completed([pool.submit(dense_job,j)for j in jobs]):print(f.result(),flush=True)
