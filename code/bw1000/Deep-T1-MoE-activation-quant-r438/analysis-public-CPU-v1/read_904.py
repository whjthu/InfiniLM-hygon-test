"""Original CPU904 entry closure. Stdlib only; no Native/device imports."""
from pathlib import Path
import argparse,base64,hashlib,json
MF='fa63f89b81492e8f5908939d79e7f3357b2f80530656866cc3cb324ef315c827'
RECEIPT='2a327d7717be014ae662289cfd5759bb6e4c6c235ee5f21f70f73074a3a25ad2'
sha=lambda b:hashlib.sha256(b).hexdigest()
def audit(path):
 packed=Path(path).read_bytes();a=json.loads(packed);assert a['actual_CPU']=='910407.904'and a['new_jobs']==0;files={}
 for p,v in a['files'].items():
  b=base64.b64decode(v['bytes'],validate=True);assert len(b)==v['nbytes']and sha(b)==v['SHA256'];files[p]=b
 def one(suffix):
  found=[v for p,v in files.items()if p.endswith(suffix)];assert len(found)==1;return found[0]
 raw=one('/result.json');assert sha(raw)==RECEIPT;r=json.loads(raw)
 assert r['completed']and r['passed']and r['actual_CPU']=='910407.904'and r['source_manifest_SHA256']==MF
 assert all(not r[k]for k in ['Native_dlopen','Runtime_execution','Model_loaded','GPU','new_HOST_TUs','new_GPU_TUs','GPU_GO','Model_GO'])
 assert len(r['affinity'])==4
 source=one('/SOURCE_MANIFEST.json');assert sha(source)==MF
 local=Path(__file__).resolve().parents[1]/'model-public-pair-v1'
 assert (local/'SOURCE_MANIFEST.json').read_bytes()==source
 for n,h in json.loads(source)['files_SHA256'].items():assert sha((local/n).read_bytes())==h,n
 expected=['actual-private-worker-eight-cases','actual49object-derivation-NO-dlopen','actual-private-CLI-NOT-serving']
 assert [c['name']for c in r['commands']]==expected
 for c in r['commands']:
  assert c['rc']==0
  for ext in ['stdout','stderr']:assert sha(one('/'+c['name']+'.'+ext))==c[ext+'_SHA256']
 worker=json.loads(one('/'+expected[0]+'.stdout'));assert worker==r['worker_control']and worker['passed']and worker['recording_CABI_only']and not worker['actual_Model_or_GPU_or_HTTP']
 cases=worker['cases'];assert [c['scenario']for c in cases]==['six-pass','bad-history','extra-request','missing-hit','reset-entry-fail','forward-fail','reset-drain-fail','canceled']and all(c['passed']for c in cases)
 assert cases[0]['mode_set_calls']==[0,1,0,1,1,0]and cases[0]['recording_Native_calls']==66
 assert cases[1]['recording_Native_calls']==0 and cases[2]['recording_Native_calls']==66 and cases[3]['full_requests_seen']==2
 assert cases[4]['reset_calls']==1 and cases[6]['reset_calls']==2
 binding=json.loads(one('/'+expected[1]+'.stdout'));assert binding==r['actual_binding']==dict(library_SHA256='e3e39de9524af7c19f6bf757236e3288c4bd7647909b2ad14f3e3ebf6d1426e9',objects=49,guard_retained=True,Native_dlopen=False)
 cli=json.loads(one('/'+expected[2]+'.stdout'));assert cli==r['actual_whole_CLI']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 assert any(x.startswith('910407.904|')and '|COMPLETED|0:0|'in x for x in a['account'])
 return dict(offline_audit_passed=True,actual_CPU='910407.904 COMPLETED0/2s',archive_SHA256=sha(packed),archive_bytes=len(packed),archive_files=len(files),receipt_SHA256=RECEIPT,source_manifest_SHA256=MF,commands=expected,actual_worker_cases=cases,actual_binding=binding,actual_whole_CLI=cli,work_wall_s=r['wall_s'],HOST_compiles=0,GPU_compiles=0,Native_dlopen=False,Runtime_execution=False,Model_loaded=False,actual_HTTP=False,performance='UNMEASURED',GPU_GO=False,Model_GO=False,limits='Eight actual changed serve/original generate/reset method scenarios use recording Native/HTTP leaves. Exact49object binder and same CLI nonserve were actually executed; no actual Model create/forward/quality or device gains.',next='Specific immutable6SSE deployment/client/controller and scope required before fresh single8card Model pair; do not repeat904/902 or old stable suites.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps({k:q[k]for k in ['offline_audit_passed','actual_CPU','archive_SHA256','Model_loaded','performance']}))
