from pathlib import Path
import base64,hashlib,json,subprocess
D=Path(__file__).parent;B=D.parent;O=B/'training-minicpm-original143-values-CPU-v1';ns={'__file__':str(D/'reuse.py')};pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
remote=r'''from pathlib import Path
import base64,hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results');p=R/'training-minicpm-original143-values-CPU-v1/prelaunch-fence.json';raw=p.read_bytes();assert len(raw)<5000
W=R/'training-minicpm-original143-values-CPU-v2';rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if '|'+W.name+'|' in x]
print(json.dumps(dict(actual_remote_fence_bytes=len(raw),actual_remote_fence_SHA256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode(),new_scope_exists=W.exists(),new_scope_accounting=rows)))
'''
with (D/'actual-fence-read-intent.json').open('x') as f:json.dump({'scope':'Exact small prelaunch guard JSON bytes only, no tensors or job'},f)
r=subprocess.run(ns['ssh'],input=remote.encode(),capture_output=True,timeout=35);assert r.returncode==0
v=json.loads(r.stdout);raw=base64.b64decode(v.pop('b64'),validate=True);assert hashlib.sha256(raw).hexdigest()==v['actual_remote_fence_SHA256']
with (D/'actual-remote-prelaunch-fence.json').open('xb') as f:f.write(raw)
local=(O/'original-prelaunch-fence.json').read_bytes();v.update(local_record_SHA256=hashlib.sha256(local).hexdigest(),local_record_CRLF=local.count(b'\r\n'),remote_original_CRLF=raw.count(b'\r\n'),semantic_equal=json.loads(local)==json.loads(raw))
assert v['semantic_equal'] and v['local_record_CRLF']>0 and v['remote_original_CRLF']==0 and v['actual_remote_fence_SHA256']!=v['local_record_SHA256'] and not v['new_scope_exists'] and not v['new_scope_accounting']
with (D/'actual-fence-byte-reconciliation.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
