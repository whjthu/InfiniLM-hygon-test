"""Read original832 bytes and frozen-source identity; stdlib only."""
from pathlib import Path
import argparse,hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
 d=Path(__file__).resolve().parent;x=json.loads(Path(path).read_bytes());r=x['receipt'];raw=r['rawtext'].encode();assert len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['SHA256']=='40ba87c2079bf28eb8581e355f30eeffffc0adc3d9c5a81074d473e46e7d5b04'
 q=json.loads(raw);assert q['completed']and q['passed']and q['actual_CPU']=='910407.832'and q['GPU']==q['Model']==0 and not any(q[k]for k in ['Native_dlopen','checkpoint_IO','compile','CPU804_820_850_repeated'])
 assert any(s=='910407.832|GLM-r414-changed-window-CPU|COMPLETED|0:0|00:00:00'for s in x['Slurm'])
 assert sha(d/'SOURCE_MANIFEST.json')==q['source_manifest_SHA256']and sha(d/'owner/SOURCE_MANIFEST.json')==q['owner_manifest_SHA256']
 for n,h in json.loads((d/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(d/n)==h,n
 rows=q['recorded16owner_window_handoffs'];assert len(rows)==96 and {(r['stage'],r['rank'],r['arm'])for r in rows}=={(s,r,a)for s in range(2)for r in range(8)for a in range(6)}
 assert all(r['window']==[2,4,2,4,4,2][r['arm']]and r['prefill_calls']==20 and r['final_past']==9741 and r['sample_count']==1 and not r['Native_executed']for r in rows)
 cli=q['actual_final_CLI'];assert cli['source_preflight_passed']and cli['Slurm']==cli['GPU']==0 and not cli['intent_created']and not cli['Native_dlopen']
 shell=cli['shell_contract'];assert shell['passed']and shell['actual_Bash']and shell['actual_production_typed_purelayout']and len(shell['roles'])==2
 assert q['cancel_after_complete_W4group']and q['exception_discard_without_replay']
 return dict(passed=True,actual_CPU=q['actual_CPU'],receipt_path=r['path'],receipt_SHA256=r['SHA256'],source_manifest_SHA256=q['source_manifest_SHA256'],owner_manifest_SHA256=q['owner_manifest_SHA256'],handoffs=96,actual_final_CLI_Bash_roles=2,consumer_s=q['elapsed_s'],GPU=0,Model=0,Native_dlopen=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('observed');a=p.parse_args();print(json.dumps(read(a.observed),indent=2))
