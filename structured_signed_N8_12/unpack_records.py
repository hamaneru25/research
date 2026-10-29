"""Expand the compressed completed-record archive for inspection or resuming."""
import gzip,json
from pathlib import Path

if __name__=='__main__':
    out=Path(__file__).parent/'results'
    data=json.loads(gzip.decompress((out/'search_records.json.gz').read_bytes()))
    for name,record in data['records'].items():
        if Path(name).name!=name or not name.endswith('.json'):raise ValueError(name)
        path=out/name
        if path.exists()and json.loads(path.read_text())!=record:raise FileExistsError(path)
        path.write_text(json.dumps(record,indent=2)+'\n')
    print('Expanded',len(data['records']),'completed records')
