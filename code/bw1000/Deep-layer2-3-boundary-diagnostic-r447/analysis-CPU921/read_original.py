"""Stdlib audit of original919/921 receipts, MD and ELF text; no subprocess."""
import argparse,hashlib,json,re,shlex
from pathlib import Path
H=lambda raw:hashlib.sha256(raw).hexdigest()
S=Path(__file__).resolve().parent.parent/'cpu-v2'

def deps(text):return set(shlex.split(text.replace(chr(92)+chr(10),' ').split(':',1)[1]))
def symbols(text):return {s.split()[-1]:s.split()[-2]for s in text.splitlines()if len(s.split())>=3}
def needed(text):return sorted(x.split('[',1)[1].split(']',1)[0]for x in text.splitlines()if'NEEDED'in x)

def audit(observation,original919):
 raw=Path(observation).read_bytes();obs=json.loads(raw);files={}
 for item in obs['files']:
  data=item['text'].encode();assert len(data)==item['bytes']and H(data)==item['SHA256']
  name=Path(item['path']).name;assert name not in files;files[name]=(data,item)
 assert obs['accounting']==[['910407.921','Deep-layer2-3-boundary-r447-CPU-v2','COMPLETED','0:0','00:00:14','a14r2n20']]
 data,ref=files['CPU-build.json'];q=json.loads(data)
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.921'
 assert q['publication_commit']=='35b032a53953acbfd5fb09f91da22515149d75b9'
 assert q['source_manifest_SHA256']==H((S/'SOURCE_MANIFEST.json').read_bytes())=='d384669c5a1ae29d8b0911b8766868a6564376a7aad0150b2f00df938f4423c6'
 source=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(H((S/p).read_bytes())==h for p,h in source['files_SHA256'].items())
 assert q['GPU']==0 and not q['Native_dlopen']and not q['Runtime_created']and not q['SDK_executed']and not q['weights_payload_opened']
 assert q['production_HOST_compiles']==q['pure_CPU_control_HOST_compiles']==1 and q['HIP_compiles']==0
 reuse=json.loads((S/'REUSE_CPU919_RECORDING.json').read_bytes());oldraw=Path(original919).read_bytes();assert H(oldraw)==reuse['receipt_SHA256']=='a8a0aeb50bfc959c405e0fe0033289828af53e91ad20c87267af20a901c39b4b'
 old=json.loads(oldraw);assert old['completed']and not old['passed']and old['actual_CPU']=='910407.919'
 local=old['actual_changed_consumer'];assert q['reused_changed_consumer_from_CPU919']==local and q['changed_recording_executed_this_job']is False
 assert local['passed']and not local['actual_Native_execution']and not local['actual_model_activations']
 assert[(c['name'],c['completed_forwards'],c['attempts'],c['resets'])for c in local['cases']]==[('normal',184,184,2),('forward_nonzero',11,12,2),('continuity_difference',91,91,1)]
 pin=json.loads((S/'PARENT_PINS.json').read_bytes());inputs=json.loads((S/'BUILD_INPUTS.json').read_bytes())
 history={r['lookup_path']:r['SHA256']for r in inputs['project_dependency_identity_records']}
 for item in inputs['original_domain_override_records']:
  assert history[item['path']]==item['PD_modified_SHA256']and H((S/item['original_snapshot']).read_bytes())==item['original_SHA256']
  history[item['path']]=item['original_SHA256']
 assert len(inputs['original_domain_override_records'])==8 and sum(r['friend_declarations_added']for r in inputs['original_domain_override_records'])==12
 assert len(q['compiled'])==2
 for c in q['compiled']:
  md,mdref=files[Path(c['MD']).name];assert H(md)==c['MD_SHA256']and deps(md.decode())==set(c['dependencies_SHA256'])and c['fresh_M_equals_MD']
  for path,sha in c['dependencies_SHA256'].items():
   if path.startswith('/refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447/cpu-v2/'):
    assert H((S/Path(path).relative_to('/refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447/cpu-v2')).read_bytes())==sha
   elif path.startswith('/refactor/'):assert history[path]==sha
 assert q['Native_object_count']==49 and q['reused_objects']==48
 expected=dict(pin['objects']);del expected[pin['replace_object']]
 actual=q['all_objects_SHA256'];assert all(actual[p]==h for p,h in expected.items())
 new={p:h for p,h in actual.items()if p not in expected};assert len(new)==1 and next(iter(new.values()))==q['compiled'][0]['object_SHA256']
 oldsyms=symbols(files['parent911-defined.log'][0].decode());newsyms=symbols(files['r447-defined.log'][0].decode());weak={'W','w','V','v','u'}
 strong={n for n,k in oldsyms.items()if k not in weak};public={n for n in oldsyms if not n.startswith('_Z')};assert strong<=set(newsyms)and public<=set(newsyms)
 required=['deep_boundary_'+n+'_r447'for n in ['arm','seal','abandon','clear','count','info','copy']];assert q['new_CABI']==required and all(n in newsyms for n in required)
 assert q['nonpublic_weak_removed_diagnostic']==sorted(set(oldsyms)-set(newsyms))
 oldneeded=needed(files['parent911-dynamic.log'][0].decode());newneeded=needed(files['r447-dynamic.log'][0].decode());assert oldneeded==newneeded==q['DT_NEEDED']
 ld=files['r447-ldd-r-NO-initializers.log'][0].decode();assert'not found'not in ld and'undefined symbol'not in ld
 assert set(needed(files['pure-control-DT_NEEDED.log'][0].decode()))<={'libstdc++.so.6','libgcc_s.so.1','libc.so.6','libm.so.6','libpthread.so.0'}
 control=json.loads(files['actual-header-control-no-Runtime.log'][0].decode().splitlines()[-1]);assert control==q['actual_header_control_recording']and control['passed']and not control['Runtime_created']and control['GPU']==0
 assert q['library_SHA256']=='8b1d1dacee6ba4b4f4d08dd38aa46c7bc529e5d3d89dc1ab753c204fe1b8084b'
 return dict(passed=True,scope='Original CPU921 compilation/link/CPU-byte-control and CPU919 recorded-consumer phase only; no actual Model activation',
  actual_CPU='910407.921 COMPLETED0/14s',producer_wall_seconds=q['seconds'],observation_SHA256=H(raw),receipt_SHA256=H(data),receipt_bytes=len(data),
  original919_FAILED_retained=True,original919_receipt_SHA256=H(oldraw),recording3cases_replayed=False,recording_cases=local['cases'],
  source_manifest_SHA256=q['source_manifest_SHA256'],source_members_verified=len(source['files_SHA256']),original_domain_headers=8,PD_friend_declarations_diagnostic=12,
  compiled_HOST_production=1,compiled_HOST_CPU_control=1,HIP_compiles=0,actual_MD_checked=2,objects_total=49,objects_reused=48,
  library=q['library'],library_SHA256=q['library_SHA256'],all_old_strong_retained=True,old_strong_count=len(strong),all_old_C_retained=True,old_C_count=len(public),
  new_CABI=required,nonpublic_weak_removed=q['nonpublic_weak_removed_diagnostic'],DT_NEEDED_unchanged=True,ldd_r_resolved=True,
  actual_Runtime_or_Native_executed=False,actual_Model_activations_captured=False,GPU_GO=False,Model_GO=False,root_cause='UNKNOWN',
  original1374_quality_FAIL_unchanged=True,raw_files_verified=len(files),
  limitation='Original producer hashes bind binaries/retained objects/SDK; this offline reader independently parses original MD and ELF text but does not download or rehash remote ELF binaries.')

def main():
 p=argparse.ArgumentParser();p.add_argument('observation',type=Path);p.add_argument('--original919',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 q=audit(a.observation,a.original919 or a.observation.parent/'Deep447-CPU919-original-result.json');text=json.dumps(q,indent=2)+'\n'
 if a.output:a.output.write_text(text,encoding='utf-8')
 print(text,end='')
if __name__=='__main__':main()
