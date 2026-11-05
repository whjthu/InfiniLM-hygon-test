from pathlib import Path
import argparse,hashlib,json,shlex
H=lambda b:hashlib.sha256(b).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--output',required=True);p.add_argument('--source',default=str(Path(__file__).resolve().parent.parent));a=p.parse_args()
d=json.loads(Path(a.original).read_bytes());S=Path(a.source)
for name,f in d['files'].items():assert len(f['text'].encode())==f['bytes']and H(f['text'].encode())==f['SHA256'],name
q=json.loads(next(f['text']for n,f in d['files'].items()if n.endswith('/CPU-result.json')))
assert q['actual_CPU']=='910407.1063'and q['completed']and q['passed']and q['CPU_only']and q['GPU']==0
assert not q['Model']and not q['Native_dlopen']and not q['SDK_device_execution']and not q['Torch_imported']and not q['project_maps']
assert q['new_HOST_TUs']==1 and q['new_GPU_TUs']==0 and d['accounting_rc']==0 and len(d['accounting'])==1 and '|COMPLETED|0:0|00:00:06|'in d['accounting'][0]
m=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert H((S/'SOURCE_MANIFEST.json').read_bytes())==q['source_manifest_SHA256']=='4529b21d88b96b86d44a10d6ba91a4d44c9e8861889160aa14b5b6f4dfeddf7e'
for n,h in m['files_SHA256'].items():assert H((S/n).read_bytes())==h,n
for name,v in q['logs'].items():
 f=next(f for p,f in d['files'].items()if p.endswith('/'+name+'.log'));assert f['SHA256']==v['SHA256']and f['bytes']==v['bytes']and v['rc']==0
log=next(f for p,f in d['files'].items()if p.endswith('/relocation-trace-no-entry.log'))
assert 'not found'not in log['text']and 'undefined symbol'not in log['text']
for n,f in d['artifact_files'].items():
 if 'text'in f:assert len(f['text'].encode())==f['bytes']and H(f['text'].encode())==f['SHA256']
obj=next(f for n,f in d['artifact_files'].items()if n.endswith('.o'));lib=next(f for n,f in d['artifact_files'].items()if n.endswith('.so'))
assert obj['SHA256']==q['compiled']['object_SHA256']and lib['SHA256']==q['library_SHA256']
md=[f for n,f in d['artifact_files'].items()if n.endswith('.d')];assert len(md)==2 and md[0]['SHA256']==md[1]['SHA256']==q['compiled']['before_MD_SHA256']==q['compiled']['actual_MD_SHA256']
for f in md:assert set(shlex.split(f['text'].replace('\\\n',' ').split(':',1)[1]))==set(q['compiled']['all_MD'])
assert q['compiled']['unchanged_before_after']and q['compiled']['source_SHA256']==H((S/'component.cpp').read_bytes())
c=json.loads((S/'BUILD_CONTRACT.json').read_bytes());assert q['exact_dependency_libraries']==c['libraries_SHA256']and q['backend_source_pins']==c['source_pins']
r=q['recording_consumer'];assert r['passed']and not r['GPU']and not r['Native_dlopen']and not r['SDK_execution']and r['reused_1062_negative_boundaries_not_rerun']
assert not q['reused_1062_negative_boundaries']['executed_again']
f=r['success']['driver'];assert f['passed']and f['cleanup_return']==0 and f['counts']==[3,3,3,3]and len(f['records'])==12 and len(f['inspections'])==4
assert all(v['values'][19:]==[1]*6 for v in f['inspections'])
out=dict(passed=True,actual_CPU=q['actual_CPU'],accounting=d['accounting'],raw_files=len(d['files']),artifacts=len(d['artifact_files']),MD_headers=len(q['compiled']['all_MD']),one_changed_HOST=True,new_GPU_TUs=0,library=q['library'],library_SHA256=q['library_SHA256'],CAPI=q['CAPI'],new_inspection_seam_recording=True,recording_GEMMs=12,recording_inspections=4,real_SDK_Runtime_numeric=False,reused_1062_negative_not_reexecuted=True,full_relocation_stdout_stderr_SHA256=log['SHA256'],GPU_GO=False,Model_GO=False)
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
