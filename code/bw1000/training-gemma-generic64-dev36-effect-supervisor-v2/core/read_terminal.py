"""Exact original four prediction files and operational receipt, never a device query or gold."""
import argparse
import json
from pathlib import Path
from effect_contract import *

def consume(operational,operational_pin,output):
    source=verify('READ')
    opraw=read(operational,operational_pin);op=json.loads(opraw)
    need(Path(operational)==ROOT/'operational-independent-terminal.json','exact new operational terminal path required')
    need(op['request']==REQUEST and op['source_manifest']==source and op['accounting_C0'] is True and
         op['owned_cleanup_passed'] is True and op['post_scope_satisfied'] is True and
         op['strong_same_host_boot_owner_terminal'] is True and op['dev_gold_read'] is False,
         'real independent device/accounting/owned terminal required, no inferred clean')
    generation=ROOT/'generated'
    final_raw=read(generation/'generation-finalized.json',op['generation_finalized'])
    final=json.loads(final_raw);sealed={};id_order=None
    result=dict(state='GEMMA_DEV36_108_INDEPENDENT_TERMINAL_V1',request=REQUEST,read_request=READ_REQUEST,
        source_manifest=source,operational_terminal=pin(opraw),planned_each=36,planned_total=108,
        passed=False,independent_passed=False,dev_gold_read=False,blind_read=False,Model_GO=False,formal17=False)
    try:
        need(final['request']==REQUEST and final['source_manifest']==source and final['epoch']==op['epoch'] and
             final['deadline']==op['deadline'] and final['passed'] is final['all108sealed_before_gold'] is
             final['all_prediction_files_closed'] is True and final['model_loads']==1 and final['dev_gold_read'] is False and
             final['blind_read'] is False and final['base_is_actual851_initial_LoRA'] is True and final['generation_cap']==256,
             'actual108 whole seal/one base load required')
        arms={}
        for arm in ARMS:
            entry=final['arms'][arm];name=arm+'-predictions-private.jsonl'
            need(entry['planned']==entry['attempted']==entry['completed']==36 and entry['missing']==0 and
                 entry['actual476_pre_forward_equal'] is True and entry['file']['name']==name,'actual arm36/476 proof required')
            raw=read(generation/name,entry['file'],cap=4<<20);rows=[json.loads(x) for x in raw.splitlines()]
            order=[r['id'] for r in rows]
            need(len(order)==len(set(order))==36 and (id_order is None or id_order==order),'same ordered unique36 in all arms required')
            id_order=order
            for i,row in enumerate(rows):
                need(row['arm']==arm and row['ordinal']==i and row['status']=='COMPLETED' and
                     all(type(row[k]) is bool for k in ('natural_EOS','cap_hit')) and
                     eos(row['raw_generated_token_ids'],row['raw_EOS_ids'])=={k:row[k] for k in
                         ('natural_EOS','cap_hit','generated_tokens','generated_ids_SHA256')},'actual pre-padding EOS provenance differs')
            sealed[name]=pin(raw)
            arms[arm]=dict(completed=36,missing=0,natural_EOS=sum(r['natural_EOS'] for r in rows),cap=sum(r['cap_hit'] for r in rows))
        sealed['generation-finalized.json']=pin(final_raw)
        result.update(passed=True,independent_passed=True,all108sealed_before_gold=True,arms=arms,sealed_predictions=sealed,
            actual_initial_named_SHA256=NAMED,initial_same_as_actual851=True,actual_updates=0,
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
