"""Read all existing failure dispositions after physically independent cleanup, without gold."""
import json
import sys

from contract import ASSET, CORE, REQUEST, need, stable_json


def consume_failure(native,source,epoch,steps):
    sys.path.insert(0,str(CORE))
    from asset_witness import INPUT_PIN
    from blind_inputs import TARGET, check_bundle
    from pathlib import Path
    from slots import verify_slots

    need(native['request_id']==REQUEST and native['epoch']==epoch and len(steps)==3
         and native['returncode']!=0 and native['cleanup_errors']==[]
         and native['held_cleanup']['reaped'] is True and native['tree_cleanup']['owned_processes_remaining']==0,
         'exact failed native child terminal and owned cleanup required')
    generation=ASSET/'generated'
    final,final_pin=stable_json(generation/'generation-finalized.json',native['generation_finalized'])
    need(final['request']==REQUEST and final['epoch']==epoch and final['source_manifest']!=source
         and final['blind_gold_read'] is final['dev_inputs_read'] is False and final['passed'] is False
         and final['all108sealed_before_gold'] is final['all_prediction_files_closed'] is True,
         'exact failed terminal has no gold and retains seal')
    from contract import core_source
    need(final['source_manifest']==core_source(),'exact failed core source required')
    rows,pin=check_bundle(Path(TARGET))
    need(pin==final['input_manifest']==INPUT_PIN,'exact original answer-free input')
    _,members,counts=verify_slots(generation,final,[row['id'] for row in rows])
    totals={kind:sum(value[kind] for value in counts.values()) for kind in ('completed','error','missing')}
    need(sum(totals.values())==108,'exact full disposition denominator')
    first=final.get('first_slot')
    if not first or first.get('state')=='FIRST_NATIVE_SLOT_FAILED':
        need(totals['completed']==0 and totals['error'] in (0,1)
             and totals['missing']==108-totals['error']
             and (not first or totals['error']==1),'failed/unreached first slot cannot continue')
    return dict(planned=108,**totals,first_state=first['state'] if first else 'NATIVE_FIRST_SLOT_NOT_ENTERED',
                failed_generation_finalized=final_pin,slot_members=members,mapped_identity_pass='miopen_infer_process_bind' in final,
                all108sealed_before_gold=True,score_GO=False,blind_gold_read=False)
