"""Produce a read-only scalar comparison of original .1139 and resumed .3650."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from types import ModuleType


R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
BASE = R/'results/training-zimage-style18-native25-resume20-v3'
STAGE = R/'incoming/t07-style18-resume10-b90028e67ec3-v2'
OUT = STAGE/'work-once/resume10'
REQUEST = 'b90028e67ec347cd8c3beb72fc0c4aa1'
LABEL = 't07-style18-resume10-b90028e67ec3-v2'
POLICY_SHA = 'a6d3da45aa8b13ba27971d6d289fa555a6ce9e4fea21f7c7df128c9d5c128209'
RECEIPT_SHA = '17ea567c6ba5f550b3974524340bb62f5b7a1e090c39ca14db3afeb8a4b07d01'
INDEPENDENT_SHA = 'bf077995f77b7ffcdbdf61df74ff152ded95fa5e26d039ab6a5bb647a76cdc85'
JOURNAL_SHA = 'dae99f14cd4a6f71d7f3f6420ca7f6d9bb46bca923bf5dd9afa373f82ce0d530'
STEPS = tuple(range(11,26))
TOL = 1e-6
FIELDS = {'input_latent_SHA256','noise_SHA256','timestep_SHA256','timestep_id'}


def need(test, code):
    if not test:
        raise ValueError(code)


def pin(raw):
    return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def policy(path):
    need(path.resolve(strict=True)==path,'EXACT_CANONICAL_POLICY_PATH')
    with path.open('rb') as stream:
        raw=stream.read(16385)
    need(pin(raw)['SHA256']==POLICY_SHA,'EXACT_FROZEN_V2_POLICY_BYTES')
    module=ModuleType('frozen_3650_policy')
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def bound_summary(obj):
    gp,post=(obj[k] for k in ('actual_GPU_step','actual_post_step'))
    cmp=obj['comparison']
    need(obj['request']==REQUEST and len(gp)==len(post)==9 and gp[:4]==['1103606.3650',LABEL+'-gpu','COMPLETED','0:0']
         and post[:4]==['1103606.3651',LABEL+'-post','COMPLETED','0:0'] and gp[5]==post[5]=='f13r2n07'
         and gp[6]=='cpu=8,gres/dcu=1,mem=96G,node=1' and post[6]=='cpu=2,mem=4G,node=1'
         and gp[8]<=post[7] and obj['post_clean'] is True and obj['planned_updates']==obj['completed_updates']==15
         and obj['missing_updates']==0 and obj['passed'] is False and cmp['paired_steps']==list(STEPS)
         and cmp['paired_native_operands']==15 and cmp['first_cursor']==10 and cmp['passed'] is False
         and cmp['resumed_optimizer_journal']==dict(bytes=1500,SHA256=JOURNAL_SHA),
         'EXACT_ORIGINAL_3650_TERMINAL_SCOPE')
    return cmp


def local_receipt(path):
    with path.open('rb') as stream:
        raw=stream.read(4542)
    need(pin(raw)==dict(bytes=4541,SHA256=RECEIPT_SHA),'EXACT_AUTHENTICATED_3650_RECEIPT')
    outer=json.loads(raw)
    r=outer['response']
    need(outer['authority_verified'] is outer['payload_flushed'] is outer['remote_received'] is True
         and outer['rc']==0 and r['scope']==REQUEST and r['independent_receipt_SHA256']==INDEPENDENT_SHA
         and r['journal_SHA256']==JOURNAL_SHA and r['native_returned_update_ordinals']==list(STEPS),
         'EXACT_AUTHENTICATED_ORIGINAL_TERMINAL_EVIDENCE')
    bound_summary(r['independent'])
    return dict(terminal_receipt=pin(raw),bound_target_independent_receipt_SHA256=INDEPENDENT_SHA,
                exact_step='1103606.3650',planned_pairs=15,stepwise_values_observed=False)


def step_pairs(left,right):
    need([row['global_step'] for row in left]==list(STEPS)==[row['global_step'] for row in right],
         'EXACT_ORDERED_STEPS_11_TO_25')
    result=[]
    for step,a,b in zip(STEPS,left,right,strict=True):
        need(a['ordinal']==b['ordinal']==(step-1)%12,'EXACT_ORIGINAL_INPUT_ORDINAL')
        for row in (a,b):
            op=row['operands']
            need(set(op)==FIELDS and type(op['timestep_id']) is int and op['timestep_id']>=0
                 and all(re.fullmatch(r'[0-9a-f]{64}',op[k]) for k in FIELDS-{'timestep_id'}),
                 'EXACT_RECORDED_INPUT_AND_DRAW_FINGERPRINTS')
            for field in ('loss','preclip_norm'):
                val=row[field]
                need(type(val) in (float,int) and isfinite(val) and (val>=0 if field=='loss' else val>0),
                     'FINITE_NATIVE_LOSS_AND_POSITIVE_PRECLIP_NORM')
        need(a['operands']==b['operands'],'ORIGINAL_RECORDED_INPUT_OR_OPERAND_MISMATCH')
        operands=json.dumps(a['operands'],sort_keys=True,separators=(',',':')).encode()
        result.append(dict(step=step,source_ordinal=a['ordinal'],equal_input_operands_SHA256=pin(operands)['SHA256'],
            continuous_loss=a['loss'],resumed_loss=b['loss'],loss_absolute_difference=abs(a['loss']-b['loss']),
            continuous_preclip_norm=a['preclip_norm'],resumed_preclip_norm=b['preclip_norm'],
            preclip_norm_absolute_difference=abs(a['preclip_norm']-b['preclip_norm'])))
    return result


def first(pairs, field, threshold):
    return next((r['step'] for r in pairs if r[field]>threshold),None)


def observation(pairs):
    numbers={}
    for field in ('loss','preclip_norm'):
        diff=field+'_absolute_difference'
        numbers[field]=dict(first_exact_nonzero_difference_step=first(pairs,diff,0),
                            first_difference_above_1e_6_step=first(pairs,diff,TOL),
                            maximum_absolute_difference=max(r[diff] for r in pairs),
                            step11_exact_nonzero=pairs[0][diff]!=0,
                            step11_above_1e_6=pairs[0][diff]>TOL)
    exceeds=numbers['loss']['first_difference_above_1e_6_step']
    return numbers,('AT_FIRST_RESUMED_FORWARD_BEFORE_FIRST_RESUMED_OPTIMIZER_UPDATE' if exceeds==11 else
                    'AFTER_AT_LEAST_ONE_RESUMED_OPTIMIZER_UPDATE' if exceeds else 'NO_RECORDED_LOSS_DIVERGENCE_ABOVE_1E_6')


def run():
    p=policy(STAGE/'resume10_policy.py')
    _,checkpoints,source=p.original_reference(BASE)
    indep=p.exact(OUT.parent/'independent-terminal.json',16384,INDEPENDENT_SHA)
    summary=bound_summary(json.loads(indep))
    resumed_journal=p.journal(OUT/'optimizer-journal.jsonl',STEPS,JOURNAL_SHA)
    restored=p.exact(OUT/'restore.json',65536)
    record=json.loads(restored)
    folded=json.loads(p.exact(OUT/'contract.json',131072))
    need(record['restored_boundary_exact'] is True and record['metadata']['global_step']==10
         and record['next']['global_step']==11 and record['next']['sample_cursor']==10
         and dict(bytes=checkpoints[0]['bytes'],SHA256=record['checkpoint_SHA256'])==p.CP10_PIN
         and p.original_contract(BASE,folded)==record['recipe_pins'], 'EXACT_ORIGINAL_CP10_RESTORE_AND_CONTRACT')
    right_raw=p.exact(OUT/'steps.jsonl',65536)
    pairs=step_pairs(source[10:],[json.loads(line) for line in right_raw.splitlines()])
    stats,boundary=observation(pairs)
    need(stats['loss']['maximum_absolute_difference']==summary['max_native_loss_difference'],
         'PER_STEP_LOSS_MUST_REPRODUCE_ORIGINAL_INDEPENDENT_MAX')
    return dict(request=REQUEST,continuous_step='1087656.1139',resumed_step='1103606.3650',
        original_local_terminal_receipt_SHA256=RECEIPT_SHA,target_independent_receipt=pin(indep),
        cp10_pin=p.CP10_PIN,restore_receipt=pin(restored),original_steps_SHA256=p.ORIGINALS['steps.jsonl'],
        resumed_steps=pin(right_raw),resumed_optimizer_journal=resumed_journal,paired_steps=len(pairs),
        first_cursor=10,per_step=pairs,divergence=stats,first_loss_above_tolerance_boundary=boundary,
        checkpoint_payload_bytes_read=0,torch_imports=0,gpu_calls=0,optimizer_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-local-terminal',type=Path)
    args=parser.parse_args()
    result=local_receipt(args.verify_local_terminal) if args.verify_local_terminal else run()
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
