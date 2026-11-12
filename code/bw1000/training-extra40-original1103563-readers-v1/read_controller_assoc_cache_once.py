"""Read this account's actual controller cache limits and usage; never change them."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;B=D.parent
remote=br'''
import datetime,json,subprocess
r=subprocess.run(['scontrol','show','assoc_mgr','flags=assoc','accounts=aclsylqidf','users=aclsylqidf'],capture_output=True,text=True,timeout=20)
assert len(r.stdout.encode())<1024**2 and len(r.stderr.encode())<65536
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rc=r.returncode,stdout=r.stdout,stderr=r.stderr,new_sbatch_submissions=0,limit_updates=0)))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=45)
for n,b in [('controller-assoc-cache.private.json',r.stdout),('controller-assoc-cache-stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(b)
result=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),limit_updates=0)
if r.returncode==0:
 v=json.loads(r.stdout);result.update(command_rc=v['rc'],observed_UTC=v['observed_UTC'],cache_lines=v['stdout'].splitlines())
with (D/'controller-assoc-cache-result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
