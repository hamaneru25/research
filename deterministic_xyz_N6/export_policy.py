"""Expand a compact policy ID into its graph and explicit loss/outcome actions."""
import argparse,gzip,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('policy_id');p.add_argument('--output',required=True);a=p.parse_args()
with gzip.open(Path(__file__).resolve().parent/'results/deterministic_policies.json.gz','rt') as f:d=json.load(f)
policy=next(r for r in d['policies'] if r['id']==a.policy_id)
expanded={k:v for k,v in policy.items() if k!='loss_record_ids'}
expanded['loss_records']=[d['loss_records'][i] for i in policy['loss_record_ids']]
Path(a.output).write_text(json.dumps(expanded,indent=2))
