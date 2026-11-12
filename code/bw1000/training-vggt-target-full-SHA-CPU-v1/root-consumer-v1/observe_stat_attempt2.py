"""Repeat only the exact read-only stat query after its SSH timeout."""
from pathlib import Path
import ast,hashlib,json,subprocess
D=Path(__file__).resolve().parent
tree=ast.parse((D/'inspect_terminal_stat_once.py').read_bytes());remote=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets)))
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=60)
for n,raw in [('stat-attempt2.stdout.private.json',r.stdout),('stat-attempt2.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,exact_original_query_payload_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),read_only=True,weight_file_bytes_read=0,original_full_hash_replayed=False)
if r.returncode==0:out['value']=json.loads(r.stdout)
with (D/'stat-attempt2-result.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
