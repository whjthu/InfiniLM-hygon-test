"""Fence only an absent original namespace; never change an existing run."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).parent;B=D.parent;source=(D/'receive_once.py').read_bytes();text=source.decode()
assert text.count('W.mkdir(mode=0o700)')==1 and text.index('W.mkdir(mode=0o700)')<text.index('subprocess.Popen(args,')
assert 'exist_ok=True' not in text[text.index('W.mkdir(mode=0o700)'):text.index('for name,row in FILES.items():')]
assert not any(token in text for token in ('rmtree(', 'W.rmdir(', 'shutil.move(', 'W.unlink('))
ns={'__file__':str(D/'reuse.py')};pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
remote=r'''from pathlib import Path
import datetime,hashlib,json,socket,subprocess,time
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original143-values-CPU-v1');label=W.name
def accounting():return [x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if '|'+label+'|' in x]
before=accounting();won=False
if not before:
 try:W.mkdir(mode=0o700);won=True
 except FileExistsError:pass
out=dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_claim='1689f91b663f5ca9973697c69185b26826823264',source_receiver_SHA256=SOURCE_SHA,atomic_absent_namespace_fence_won=won,accounting_before=before,accounting_after=accounting(),host=socket.gethostname(),boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),fence_monotonic_ns=time.monotonic_ns(),original_exec_transport_outcome='UNKNOWN',no_existing_directory_modified=True)
out['original_audit_cannot_start']=bool(won and not before and not out['accounting_after'])
if won:
 with (W/'prelaunch-fence.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
'''
with (D/'fence-intent.json').open('x') as f:json.dump(dict(scope='Only atomic mkdir if absent and no original Slurm label. Existing directory means no mutation. Frozen original requires mkdir before any audited worker.',source_SHA256=hashlib.sha256(source).hexdigest()),f,indent=2)
r=subprocess.run(ns['ssh'],input=('SOURCE_SHA='+repr(hashlib.sha256(source).hexdigest())+'\n'+remote).encode(),capture_output=True,timeout=40);assert r.returncode==0
v=json.loads(r.stdout)
with (D/'original-prelaunch-fence.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
