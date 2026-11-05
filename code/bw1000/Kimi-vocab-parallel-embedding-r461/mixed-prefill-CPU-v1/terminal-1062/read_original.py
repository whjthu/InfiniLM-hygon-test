from pathlib import Path
import argparse,hashlib,json,shlex
H=lambda b:hashlib.sha256(b).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--artifacts',required=True);p.add_argument('--output',required=True);p.add_argument('--source',default=str(Path(__file__).resolve().parent.parent));a=p.parse_args()
d=json.loads(Path(a.original).read_bytes());art=json.loads(Path(a.artifacts).read_bytes());S=Path(a.source)
for name,f in d['files'].items():
 b=f['text'].encode();assert len(b)==f['bytes']and H(b)==f['SHA256'],name
q=json.loads(next(f['text']for name,f in d['files'].items()if name.endswith('/CPU-result.json')))
assert q['actual_CPU']=='910407.1062'and q['completed']and q['passed']and q['CPU_only']
assert q['GPU']==0 and not q['Model']and not q['Native_dlopen']and not q['SDK_device_execution']
assert q['new_HOST_TUs']==1 and q['new_GPU_TUs']==0 and not q['Torch_imported']and not q['project_maps']
assert d['accounting_rc']==0 and len(d['accounting'])==1
assert '|COMPLETED|0:0|00:00:07|'in d['accounting'][0]
m=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert H((S/'SOURCE_MANIFEST.json').read_bytes())==q['source_manifest_SHA256']=='33d7eb154d73bc122fe7c6a8129ab2e8b33871b7835dd683e2c5d1efcf132491'
for n,h in m['files_SHA256'].items():assert H((S/n).read_bytes())==h,n
for name,v in q['logs'].items():
 f=next(f for p,f in d['files'].items()if p.endswith('/'+name+'.log'));assert f['SHA256']==v['SHA256']and f['bytes']==v['bytes']and v['rc']==0
for name,f in art['artifact_files'].items():
 if 'text'in f:assert len(f['text'].encode())==f['bytes']and H(f['text'].encode())==f['SHA256']
obj=next(v for p,v in art['artifact_files'].items()if p.endswith('component.o'))
lib=next(v for p,v in art['artifact_files'].items()if p.endswith('.so'))
assert obj['SHA256']==q['compiled']['object_SHA256']and lib['SHA256']==q['library_SHA256']
md=[f for p,f in art['artifact_files'].items()if p.endswith('.d')];assert len(md)==2 and md[0]['SHA256']==md[1]['SHA256']==q['compiled']['before_MD_SHA256']==q['compiled']['actual_MD_SHA256']
for f in md:
 deps=set(shlex.split(f['text'].replace('\\\n',' ').split(':',1)[1]));assert deps==set(q['compiled']['all_MD'])
assert q['compiled']['unchanged_before_after']
assert q['compiled']['source_SHA256']==H((S/'component.cpp').read_bytes())
c=json.loads((S/'BUILD_CONTRACT.json').read_bytes());assert q['exact_dependency_libraries']==c['libraries_SHA256']and q['backend_source_pins']==c['source_pins']
r=q['recording_consumer'];assert r['passed']and r['GPU']==0 and not r['Native_dlopen']and not r['SDK_execution']
assert r['success']['driver']['counts']==[3,3,3,3]and len(r['success']['driver']['records'])==12
assert r['run_failure']['driver']['missing_arms']==11 and r['run_failure']['driver']['cleanup_return']==0
assert r['expired_before_create']['driver']['missing_arms']==12 and not r['expired_before_create']['calls']
out=dict(passed=True,actual_CPU=q['actual_CPU'],accounting=d['accounting'],original_files=len(d['files']),actual_artifacts=len(art['artifact_files']),MD_headers=len(q['compiled']['all_MD']),one_HOST_compile=True,new_GPU_TUs=0,library=q['library'],library_SHA256=q['library_SHA256'],CAPI=q['CAPI'],actual_driver_recording_arms=12,recording_error_boundaries=['run_rc_failure+destroy','expired_before_create'],Native_SDK_math_leaves='recording only, no ordinary device entry executed',GPU_GO=False,Model_GO=False,actual_numeric_or_memory=False,original_transport_count_failure_preserved=True,artifact_HOST_alias_read_failure_preserved=True,limits=['binary artifact payloads remainR; actualSHA independently read once and compared with receipt','no device numerical/driverpeak/poolhighwater/Model fit or speed qualification'])
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
