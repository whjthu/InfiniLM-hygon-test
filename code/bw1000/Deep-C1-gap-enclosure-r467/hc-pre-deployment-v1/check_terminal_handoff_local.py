"""One preserved actual entry2789 recording -> real new terminal collector."""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,importlib,json,os,sys
D=Path(__file__).resolve().parent;P=D.parent/'hc-pre-entry-v1';sys.path[:0]=[str(D),str(P)]
import check_entry_local as producer
import collect as receiver
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ENTRY_MF='413dff01e160851b494c4f1ce2b4314e982b1909a00bf0d11faf5aecc0d8b5e4'

def run(private_input,artifacts):
 assert H(P/'SOURCE_MANIFEST.json')==ENTRY_MF
 root=Path(artifacts).resolve();assert not root.exists(),'Preserved entry artifacts exist; use existing terminal files, do not recreate'
 collected=[];directory_calls=[]
 class PreservedDirectory:
  def __init__(self,*args,**kw):
   assert not args and not kw and not directory_calls;directory_calls.append(str(root))
  def __enter__(self):root.mkdir();return str(root)
  def __exit__(self,typ,error,tb):
   # Original production entry/report/packet bytes remain here even when this
   # new receiver rejects a field. A reader fix must consume these same files.
   if error is None:
    report=json.loads((root/'run/entry-report.json').read_bytes());q=receiver.collect(root/'run',report['identity']['actual_step'],report['identity']['gpu_start_epoch']);collected.append(q)
   return False
 # Only the output-directory lifetime is extended; original producer still
 # invokes actual entry.main --execute and its existing recording leaves.
 with patch.object(producer.tempfile,'TemporaryDirectory',PreservedDirectory):proof=producer.main(private_input)
 assert proof['passed']and len(collected)==1 and len(directory_calls)==1
 q=collected[0];assert q['normal_Native_shutdown']and q['normal8owner_exit']and q['logical_requests']==5 and q['Native_CAPI_calls']==530 and q['natural_EOS']==5 and q['caps']==0
 assert q['profile_coverage_complete']and q['actual_profile_windows']==2 and len(q['scalar_event_packets'])==6
 assert set(q['state_checked_sample_CAPI_comparison'])=={'8','87'}and all(v['same_prompt_and_generated_prefix']for v in q['state_checked_sample_CAPI_comparison'].values())
 assert H(P/'SOURCE_MANIFEST.json')==ENTRY_MF
 report=root/'run/entry-report.json';files=[]
 for f in sorted(root.rglob('*')):
  if f.is_file():files.append(dict(path=str(f),bytes=f.stat().st_size,SHA256=H(f)))
 result=dict(passed=True,source_entry_manifest_SHA256=ENTRY_MF,actual_frozen_entry_main_execute=True,actual_new_terminal_collector=True,original_entry_source_unchanged=True,old_three_cases_not_rerun=True,producer_proof=proof,collector=dict(normal_shutdown=True,Native_CAPI_calls=530,requests=5,samples=[8,87],scalar_packets=6,prompt_and_generated_prefix_checked=True),original_artifacts=dict(directory=str(root),entry_report_SHA256=H(report),files=files),production_collector_SHA256=H(D/'collect.py'),scope='New terminal-schema handoff only; Native/device/weights/process/ELF leaves recording. Recording times are not device profile or performance evidence.',Native_dlopen=False,SDK_Runtime=False,GPU=False,Model_READY=False,new_jobs=0,remote_operations=0)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--artifacts',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 q=run(a.input,a.artifacts);Path(a.output).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps({k:q[k]for k in['passed','actual_frozen_entry_main_execute','actual_new_terminal_collector','collector','Native_dlopen','GPU','new_jobs']}))
