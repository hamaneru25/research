"""Read aggregate search output or the independently readable publication shards."""
from pathlib import Path
import gzip,json


def read_graph_records(root):
    path=Path(root)/'results/evaluated_graphs.json.gz'
    if path.exists():
        with gzip.open(path,'rt') as f:return json.load(f)
    paths=sorted((Path(root)/'results').glob('evaluated_graphs.part*.json.gz'))
    if not paths:raise FileNotFoundError('No evaluated graph records; run search.py')
    records=[]
    for path in paths:
        with gzip.open(path,'rt') as f:records.extend(json.load(f))
    return records


def pack_graph_records(root,chunk=800):
    records=read_graph_records(root)
    for old in (Path(root)/'results').glob('evaluated_graphs.part*.json.gz'):old.unlink()
    for i in range(0,len(records),chunk):
        path=Path(root)/'results'/f'evaluated_graphs.part{i//chunk:03d}.json.gz'
        with gzip.open(path,'wt') as f:json.dump(records[i:i+chunk],f,separators=(',',':'))
