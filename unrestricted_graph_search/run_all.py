"""Reproduce graph search, refinements, independent checks and figures."""
from pathlib import Path
import argparse,subprocess,sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true',help='Rebuild generated graph-profile caches')
    args=ap.parse_args();root=Path(__file__).resolve().parent
    if args.fresh:
        for p in (root/'results').glob('catalog_N*.json.gz'):p.unlink()
    for script in ('search_graphs.py','refine_topologies.py','local_edge_search.py',
                   'validate_engine.py','validate_winners.py','certify_four_sensors.py','make_report.py'):
        print('Running '+script,flush=True)
        subprocess.run([sys.executable,str(root/script)],cwd=root,check=True)


if __name__=='__main__':main()
