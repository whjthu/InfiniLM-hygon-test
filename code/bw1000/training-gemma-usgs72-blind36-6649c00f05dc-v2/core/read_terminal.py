"""Exact original four prediction files and operational receipt, never a device query or gold."""
import argparse
import json
from pathlib import Path
from effect_contract import *
from blind_inputs import TARGET,check_bundle
from slots import verify_slots

def consume(operational,operational_pin,output):
    source=verify('READ')
    opraw=read(operational,operational_pin);op=json.loads(opraw)
    need(Path(operational)==ROOT/'operational-independent-terminal.json','exact new operational terminal path required')
    need(op['request']==REQUEST and op['source_manifest']==source and op['accounting_C0'] is True and
         op['owned_cleanup_passed'] is True and op['post_scope_satisfied'] is True and
         op['strong_same_host_boot_owner_terminal'] is True and op['blind_gold_read'] is False,
         'real independent device/accounting/owned terminal required, no inferred clean')
    generation=ROOT/'generated'
    final_raw=read(generation/'generation-finalized.json',op['generation_finalized'])
    final=json.loads(final_raw);sealed={};id_order=None
    result=dict(state='GEMMA_BLIND36_108_INDEPENDENT_TERMINAL_V2',request=REQUEST,read_request=READ_REQUEST,
        source_manifest=source,operational_terminal=pin(opraw),planned_each=36,planned_total=108,
        passed=False,independent_passed=False,blind_gold_read=False,dev_inputs_read=False,Model_GO=False,formal17=False)
    try:
        need(final['request']==REQUEST and final['source_manifest']==source and final['epoch']==op['epoch'] and
             final['deadline']==op['deadline'] and final['binding']==op['binding'] and 'failure' not in final and final['passed'] is final['all108sealed_before_gold'] is
             final['all_prediction_files_closed'] is True and final['model_loads']==1 and final['blind_gold_read'] is False and
             final['dev_inputs_read'] is False and final['base_is_actual851_initial_LoRA'] is True and final['generation_cap']==256,
             'actual108 whole seal/one base load required')
        inputs,input_manifest=check_bundle(Path(TARGET))
        need(input_manifest==final['input_manifest'],'actual answer-free input identity differs')
        id_order=[row['id'] for row in inputs]
        _,sealed,arms=verify_slots(generation,final,id_order)
        sealed['generation-finalized.json']=pin(final_raw)
        result.update(passed=True,independent_passed=True,all108sealed_before_gold=True,arms=arms,sealed_predictions=sealed,
            input_manifest=input_manifest,actual_initial_named_SHA256=NAMED,initial_same_as_actual851=True,actual_updates=0,
            full_ID_order_SHA256=pin(json.dumps(id_order).encode())['SHA256'],owned_cleanup_passed=True,
            accounting_C0=True,post_scope_satisfied=True,allFDclean_claimed=False,exclusive_GPU_claimed=False)
    except BaseException as error:
        result.update(failure=dict(type=type(error).__name__,private_reason=str(error)[:512]),
            actual_arm_counts={arm:{k:final.get('arms',{}).get(arm,{}).get(k) for k in ('attempted','completed','missing')} for arm in ARMS})
    dump(output,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('operational','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--operational-bytes',type=int,required=True);p.add_argument('--operational-sha',required=True)
    a=p.parse_args();r=consume(a.operational,dict(bytes=a.operational_bytes,SHA256=a.operational_sha),a.output)
    print(json.dumps(r,sort_keys=True));raise SystemExit(0 if r['passed'] else 1)
