"""Consume only the saved .189 small bundle and its exact published worker source."""
import ast
import hashlib
import json
from pathlib import Path
import re

G=Path(__file__).resolve().parents[2]
P=G/'local-only/T09-execution-mount-CPU-v2-original-failure-first-read/stdout.private'
D=G/'code/bw1000/training-hunyuanvideo-execution-mount-SHA-CPU-v2'
def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())

def main():
    raw=P.read_bytes();assert pin(raw)==dict(bytes=8422,SHA256='92e03d2a22ab3c622c2606674cb1722cb5771bfc2bed12c3c49647f51a9ca40e')
    v=json.loads(raw);c=v['context'];f=v['files']
    assert c['request_id']=='487949ac4113400f9e9a96c48d248efe'
    assert c['publication']=='acd573d390e3bff20a2abe1440f4bafdf4988551'
    assert c['epoch']==1794963969.3021424
    mf=(D/'SOURCE_FILES.json').read_bytes()
    assert hashlib.sha256(mf).hexdigest()==c['source_manifest_SHA256']=='439eb2fd309fa4e7647a9c4b83fa6d4fe58b4c7008c66ceabadcd676e0b30f04'
    worker=(D/'worker.py').read_bytes();assert pin(worker)==json.loads(mf)['files']['worker.py']
    o=f['outcome.json']['value'];i=f['intent.json']['value']
    for key in ('request_id','publication','epoch','source_manifest_SHA256'):assert o[key]==i[key]==c[key]
    row=v['accounting'][0]
    assert len(v['accounting'])==1 and row==o['actual_accounting']
    assert row==['1103606.189','training-hunyuanvideo-execution-mount-SHA-CPU-v2','FAILED','1:0','0','f13r3n13','cpu=2,mem=4G,node=1']
    assert v['accounting_columns']==o['accounting_columns']==['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES']
    assert o['passed'] is False and o['current_step_outcome']=='TERMINAL' and o['worker_receipt'] is None
    assert o['held_child']['reaped'] is True and o['held_child']['returncode']==1
    assert o['held_child']['termination_needed'] is False and o['held_child']['owned_group_signaled'] is False
    assert f['worker-intent.json']==f['weight-receipt.json']==dict(exists=False)
    error=f['worker.stderr'];assert pin(error['value'].encode())==dict(bytes=error['bytes'],SHA256=error['SHA256'])
    tree=ast.parse(worker);mainfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    priority=next(n for n in ast.walk(mainfn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='need'
        and any(isinstance(x,ast.Constant) and x.value=='ACTUAL_NICE10' for x in n.args))
    hashcall=next(n for n in ast.walk(mainfn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='hash_on_execution_mount')
    intentopen=next(n for n in ast.walk(mainfn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='open'
        and any(isinstance(x,ast.Constant) and x.value=='worker-intent.json' for x in ast.walk(n.func.value)))
    assert priority.lineno==43 and priority.lineno<intentopen.lineno<hashcall.lineno
    assert 'ValueError: ACTUAL_NICE10' in error['value'] and re.search(r'line 43, in main\b',error['value'])
    assert i['original_TDB_request_state']=='UNKNOWN_NO_CANCEL_NO_RESERVATION_RELEASE'
    out=dict(schema='T09_CPU189_SAVED_BUNDLE_INDEPENDENT_FAILURE_V1',passed=True,model_or_hash_passed=False,
        context=c,original_state='SLURM_STEP_FAILED_PRE_WEIGHT_PRIORITY_GATE',actual_accounting=row,
        accounting_columns=v['accounting_columns'],actual_CPU_step_submissions=1,actual_GPU_calls=0,
        source_worker=pin(worker),source_manifest=pin(mf),saved_bundle=dict(path=str(P.relative_to(G)).replace('\\','/'),**pin(raw)),
        exception_type='ValueError',exception_reason='ACTUAL_NICE10',worker_failure_line=priority.lineno,
        worker_intent_source_line=intentopen.lineno,hash_call_source_line=hashcall.lineno,
        source_flow_proof='Actual traceback at unique priority guard; guard precedes worker intent and sole whole-hash call in exact published source. Missing outputs are supporting evidence, not the sole zero-read inference.',
        logical_weight_body_bytes_read=0,fullweight_execution_mount_qualified=False,
        inherited_nice='UNKNOWN_NOT_SAVED',actual_nice_at_failure='UNKNOWN_NOT_SAVED',
        cause='os.nice(10) applies a relative increment while the predicate incorrectly requires absolute nice10; inherited starting nice is not proven.',
        held_child=o['held_child'],original_TDB_request_state=i['original_TDB_request_state'],
        original_v1_failed_prelaunch_preserved=True,CPU_GO=False,GPU_GO=False,Model_GO=False,
        old_CPU_v2_replay_allowed=False,independent_tensor_or_body_rehash=False,remote_calls=0,
        actual_small_records={name:{key:value[key] for key in ('bytes','SHA256')} for name,value in f.items() if 'bytes' in value})
    target=Path(__file__).with_name('result.json');b=(json.dumps(out,indent=2,sort_keys=True)+'\n').encode()
    with target.open('xb') as stream:stream.write(b)
    print(json.dumps(dict(passed=True,result=str(target),**pin(b),actual_step=row[0],weight_bytes=0)))

if __name__=='__main__':main()
