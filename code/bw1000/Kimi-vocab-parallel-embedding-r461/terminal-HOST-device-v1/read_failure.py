"""Offline audit of existing bounded HOST failure evidence. No remote archive.

Reads the already-downloaded launcher/ledger/process JSON and published source.
Does not import candidate code or access SSH/Slurm/Native/SDK.
"""
from pathlib import Path
import hashlib,json,sys
H=lambda b:hashlib.sha256(b).hexdigest()
SOURCE='932cafea926f90f5c644281724e166b6965ac961818952c34001dfed7f769473'
LABEL='Kimi-r461-embedding-probe-vocab-tp32-embed-20261104-a'
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
NAMES=['Kimi461-device-v1-actual-launch-first-original.json',
 'Kimi461-device-v1-HOST-timeout-original-reconcile.json',
 'Kimi461-device-v1-HOST-owned-process-first-original.json',
 'Kimi461-device-source-upload-first-original.json','Kimi461-device-source-published.json']
def audit(workspace,repo):
 workspace=Path(workspace);repo=Path(repo);docs={};local={}
 for name in NAMES:
  b=(workspace/name).read_bytes();docs[name]=json.loads(b);local[name]=dict(bytes=len(b),SHA256=H(b))
 launch=docs[NAMES[0]];reconcile=docs[NAMES[1]];processes=docs[NAMES[2]];upload=docs[NAMES[3]]
 assert launch['invoker_PID']==launch['PGID']==1036266 and launch['returncode']==1 and not launch['timeout']and 90<=launch['elapsed_s']<110
 for k in ['stdout','stderr']:
  raw=launch[k]['rawtext'].encode();assert len(raw)==launch[k]['bytes']and H(raw)==launch[k]['SHA256']
 assert launch['stdout']['bytes']==0 and 'whole CLI90s failed'in launch['stderr']['rawtext']and 'communicate(timeout=90)'in launch['stderr']['rawtext']
 assert reconcile['invoker_directory_exists']and reconcile['accounting']['returncode']==0 and not reconcile['accounting']['stdout']
 files=reconcile['files']
 for p,v in files.items():b=v['rawtext'].encode();assert len(b)==v['bytes']and H(b)==v['SHA256'],p
 ledger=files[R+'/results/'+LABEL+'-HOST-whole-cli.json'];value=json.loads(ledger['rawtext'])
 assert value['source_manifest_SHA256']==SOURCE and value['phase']=='HOST_TIMEOUT_STOP'and value['timeout90']
 assert value['child_PID']==value['owned_PGID']==1037835 and value['returncode']==-15
 assert value['stdout']==''and value['stderr']=='Failed to collect own child after TERM/KILL'
 assert value['finished_unix']>=value['started_unix']+90
 assert R+'/results/'+LABEL+'-intent.json'not in files and R+'/results/'+LABEL+'-watch.json'not in files
 assert processes['read_only']and processes['own_UID']==154459 and processes['processes']==[]and processes['observed_unix']>value['finished_unix']
 d=repo/'code/bw1000/Kimi-vocab-parallel-embedding-r461/device-v1'
 assert H((d/'SOURCE_MANIFEST.json').read_bytes())==SOURCE
 manifest=json.loads((d/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in manifest['files_SHA256'].items():assert H((d/name).read_bytes())==h,name
 assert upload['uploaded']and len(upload['files'])==19 and upload['files']['SOURCE_MANIFEST.json']['SHA256']==SOURCE
 for name,pin in upload['files'].items():assert H((d/name).read_bytes())==pin['SHA256']
 return dict(schema=1,audit_passed=True,result_passed=False,state='HOST_WHOLE_CLI90_TIMEOUT_STOP_0INTENT_0SLURM',
  source_manifest_SHA256=SOURCE,publication_commit='f11d8f2ee5965dff5ac76a042b680ba626be2431',invoker_PID=1036266,whole_cli_child_PID=1037835,
  launcher_returncode=1,whole_cli_child_returncode=-15,original_timeout_s=90,launcher_elapsed_s=launch['elapsed_s'],
  intent_created=False,CPU_keeper_submitted=False,GPU_submitted=False,GPU=0,planned_cards_not_busy=32,Native_lookups=0,checkpoint_keys_loaded=0,
  same_UID_exact_scripts_and_recorded_PGIDs_gone_at_passive_observation=True,
  physical_floor_or_device_numerical_result='UNKNOWN: no constructor or lookup device execution was admitted',
  local_originals=local,remote_existing_originals={p:{k:v[k]for k in ['bytes','SHA256']}for p,v in files.items()},
  failed_extra_archive_auto_review_rejected_not_executed=True,extra_remote_archive_created=False,
  limitations=['The captured stderr2411B is the launcher exception trace. Child stdout/stderr was not fully collected; ledger stderr is a cleanup placeholder, not actual child diagnostic output.',
   'No container Bash/Python/config/index/header phase markers exist in this source. The timeout cause is unknown; not attributed to NFS, GIL or Native.',
   'Original namespace/one-shot HOST ledger is consumed; do not replay this launcher or extend its90s limit.',
   'Same-UID exact script and recorded PGID passive check found none. This is not a claim about every unrelated system/container helper process.'])
if __name__=='__main__':
 out=audit(sys.argv[1],sys.argv[2]);s=json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 if len(sys.argv)>3:Path(sys.argv[3]).write_text(s,encoding='utf-8',newline='\n')
 print(s,end='')
