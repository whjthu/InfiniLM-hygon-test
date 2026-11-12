"""PREPARED ONLY. Requires explicit exception to the prior no-replay rule.

Use the original namespace as an exclusive directory gate. The published
original also must create this directory before it can call sbatch. Either
actor therefore fails before submission when the other already owns the gate.
No existing directory, job, intent or result is modified or deleted.
"""
from pathlib import Path
import ast,datetime,hashlib,json,os,subprocess,time

epoch=time.time();anchor=time.monotonic();deadline=epoch+180

assert EXPLICIT_USER_RECOVERY_APPROVAL is True
assert ORIGINAL_REQUEST=='42f2a1e31ffc476d8b03cf27082a79c2'
assert ORIGINAL_CLAIM=='76cebab37bcb4669695be37b9b8021665ab7a427'
assert hashlib.sha256(ORIGINAL_SOURCE).hexdigest()==ORIGINAL_SOURCE_SHA256
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-extra40-resource-expansion-v1')
assert not W.exists(), 'Original namespace exists; readonly reconciliation required, no recovery submission'
label='training-extra40-long-20261112-v1'
for args in ([
    'squeue','-h','-u',str(os.getuid()),'-o','%i|%100j|%T'],[
    'sacct','-X','-n','-P','-u',str(os.getuid()),'-S','2026-11-12','--format=JobIDRaw,JobName%100,State']):
    assert time.time()<deadline and time.monotonic()-anchor<180
    r=subprocess.run(args,capture_output=True,text=True,timeout=min(20,max(1,deadline-time.time())))
    assert r.returncode==0 and len(r.stdout)<1024**2
    assert not any(len(row.split('|'))>=2 and row.split('|')[1].strip()==label for row in r.stdout.splitlines()), 'Matching original job exists; adopt its actual handle, no submission'

tree=ast.parse(ORIGINAL_SOURCE)
removed=[];body=[]
for node in tree.body:
    code=ast.unparse(node)
    if code in ('assert not W.exists()', 'W.mkdir(mode=448)',
                'epoch = time.time()', 'anchor = time.monotonic()', 'deadline = epoch + 180'):
        removed.append(code)
    else:body.append(node)
assert sorted(removed)==sorted(['assert not W.exists()','W.mkdir(mode=448)',
    'epoch = time.time()','anchor = time.monotonic()','deadline = epoch + 180'])
tree.body=body
# POSIX mkdir is the sole shared gate; exist_ok is deliberately not used.
# Original code has the same mandatory mkdir before its durable intent/sbatch.
W.mkdir(mode=0o700)
with (W/'recovery-intent.json').open('x') as f:
    json.dump(dict(original_request_id=ORIGINAL_REQUEST,original_claim=ORIGINAL_CLAIM,
        original_source_SHA256=ORIGINAL_SOURCE_SHA256,recovery_claim=CLAIM,recovery_request_id=REQUEST,
        explicit_user_exception=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        epoch=epoch,deadline=deadline,
        gate='Exclusive original namespace mkdir;original and recovery cannot both pass it.',
        old_jobs_cancelled=0,maximum_extra_nodes=40,new_service=False,new_poller=False),f,indent=2)
exec(compile(ast.fix_missing_locations(tree),'published_original_extra40_with_exclusive_recovery_gate','exec'),globals())
