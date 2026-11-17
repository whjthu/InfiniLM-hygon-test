"""Only new T04 pin/schema/CLI boundaries: no Torch, SSH, .pt or derivation."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent
G=D.parents[2]
P=json.loads((D/'PLAN.json').read_bytes())
valid=[];rejected=[]


def module(name):
    spec=importlib.util.spec_from_file_location(name,D/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def reject(name,call):
    try:call()
    except ValueError:rejected.append(name)
    else:raise AssertionError(name+' admitted')


def main():
    for path in D.rglob('*.py'):ast.parse(path.read_bytes())
    old=G/'code/bw1000/training-internvl-A100-input-fingerprints-CPU-v1'
    canonical=(D/'canonical/first_update_audit.py').read_bytes()
    fm=json.loads((G/'code/bw1000/training-llava-A100-first-updates-v1/FRAMEWORK_FILES.json').read_bytes())
    assert len(canonical)==fm['files']['vlm/runtime/first_update_audit.py']['bytes']==6939
    assert hashlib.sha256(canonical).hexdigest()==fm['files']['vlm/runtime/first_update_audit.py']['SHA256']==P['fingerprint_source_SHA256']
    oldpins=json.loads((G/'code/bw1000/training-llava-A100-capture-value-reader-v1/ORIGINAL_PINS.json').read_bytes())
    assert P['case']=='T04.sft' and P['reference_request_id']==oldpins['request_id']=='640a2e0f2ae4440a9689912f2de413be'
    for rank in range(8):
        name='capture/rank-%02d/receipt.json'%rank;pin=P['source_receipts'][str(rank)]
        assert pin==dict(path=P['reference_root']+'/'+name,**oldpins['originals'][name])
    valid.append('actual_T04_8receipt_pins_and_canonical350_manifest')
    C=module('input_contract')
    binding=json.loads((G/'code/bw1000/training-llava-A100-capture-value-reader-v1/LLAVA_NATIVE_BINDING.json').read_bytes())
    C.native_binding(binding);valid.append('actual_T04_native_binding_metadata')
    for key,value in (('case','T03.sft'),('attention','sdpa'),('model_class','InternVLForConditionalGeneration')):
        bad=copy.deepcopy(binding);bad['spec'][key]=value
        reject('foreign_binding_'+key,lambda bad=bad:C.native_binding(bad))
    bad=copy.deepcopy(binding);bad['spec']['training']['gradient_checkpointing']=True
    reject('binding_GC',lambda:C.native_binding(bad))
    expected={'initial.pt'}|{'forward-%02d-%s.pt'%(n,k) for n in range(1,9) for k in ('input','output')} \
        |{'update-%02d-%s.pt'%(n,k) for n in (1,2) for k in ('preclip','postclip','before-optimizer','after-optimizer')}
    receipt=dict(state='TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED',rank=0,world_size=8,completed_global_step=2,
                 optimizer_updates=2,clip_calls=2,micro_forwards=8,files=[dict(name=n,bytes=1,SHA256='a'*64) for n in sorted(expected)])
    fixture=dict(original_capture_bytes_per_rank=[25]*8)
    assert len(C.receipt_files(receipt,0,fixture))==25
    valid.append('new_receipt_schema_CPU_fixture_not_original_payload')
    for key,value in (('rank',1),('world_size',1),('completed_global_step',1),('micro_forwards',7)):
        bad=copy.deepcopy(receipt);bad[key]=value
        reject('receipt_'+key,lambda bad=bad:C.receipt_files(bad,0,fixture))
    for name,edit in (('duplicate_name',lambda rows:rows.__setitem__(0,rows[1])),
                      ('traversal',lambda rows:rows[0].__setitem__('name','../forward-01-input.pt')),
                      ('bad_SHA',lambda rows:rows[0].__setitem__('SHA256','z'*64))):
        bad=copy.deepcopy(receipt);edit(bad['files']);reject(name,lambda bad=bad:C.receipt_files(bad,0,fixture))
    row=dict(global_step=0,micro_ordinal=1)
    payload=dict(global_step=0,micro_ordinal=1,args=(),kwargs={},RNG_before_forward={})
    C.input_schema(payload,row);valid.append('new_input_projection_schema_CPU_fixture')
    for key,value in (('global_step',True),('micro_ordinal',2),('args',{}),('kwargs',[])):
        bad=dict(payload);bad[key]=value;reject('input_'+key,lambda bad=bad:C.input_schema(bad,row))
    L=module('launch_once');claim=json.loads((D/'CLAIM.json').read_bytes());L.check_claim(claim)
    assert L.cli(['--publication','a'*40,'--source-manifest-sha','b'*64,'--claim-sha256','c'*64]).publication=='a'*40
    valid.append('T04_CLI_claim')
    for key,value in (('case','T03.sft'),('InternVL_included',True),('maximum_CPU_invocations',True),('CPU_GO',False)):
        bad=dict(claim);bad[key]=value;reject('claim_'+key,lambda bad=bad:L.check_claim(bad))
    def fn(path,name):
        return ast.dump(next(n for n in ast.parse(path.read_bytes()).body if getattr(n,'name',None)==name),include_attributes=False)
    assert fn(D/'derive_inputs.py','load_original')==fn(old/'derive_inputs.py','load_original')
    assert fn(D/'common.py','Budget')==fn(old/'common.py','Budget')
    worker=(D/'derive_inputs.py').read_text();assert worker.index('rows,receipt_stamps,total=resolve_all')<worker.index('import torch')
    assert 'torch' not in sys.modules and 'swift' not in sys.modules
    print(json.dumps(dict(passed=True,valid=valid,rejected=rejected,canonical_exact=True,unchanged_load_and_Budget_AST=True,
                         Torch_imports=0,SSH_calls=0,payload_reads=0,GPU=0,actual_derivations=0)))


if __name__=='__main__':
    main()
