"""Consume the original finished Qwen math recovery branches without replay."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-math-sdpa-resume20-v1')
CONTINUOUS = ROOT.parent/'training-qwen-math-sdpa-recovery-pair-v1'/'continuous'
LABEL = ROOT.name
REQUEST = '31c1846793c5bf960011dc08cf2abdd0'
PARENT = '1087656'
NODE = 'a01r4n14'
COLUMNS = 'JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%170'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def small(path, missing=False):
    try:
        if not 0 < path.stat().st_size <= 262144:
            raise ValueError('Unexpected receipt size')
        raw = path.read_bytes()
    except FileNotFoundError:
        if missing:
            return None, None
        raise
    return json.loads(raw), dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def query(*args):
    out = subprocess.run(args, capture_output=True, timeout=12, check=True, text=True)
    return out.stdout.strip().splitlines()


def terminal_state(row):
    return len(row)==7 and row[2].split()[0] in TERMINAL


def check_native(terminal, allocation):
    valid = {row[1]:row for row in allocation}
    if set(valid)!={LABEL+'-pre',LABEL,LABEL+'-post'} or len(allocation)!=3:
        raise ValueError('Missing or duplicate original Slurm steps')
    if any(not terminal_state(row) or row[2:4]!=['COMPLETED','0:0'] or row[5]!=NODE for row in allocation):
        raise ValueError('A real Slurm step failed')
    if 'gres/dcu=8' not in valid[LABEL][6].split(','):
        raise ValueError('Incorrect number of DCUs in accounting')
    before, after = terminal['before'], terminal['after']
    if (not terminal['keeper_passed'] or not before or not after
            or before['step']!=valid[LABEL+'-pre'][0] or after['step']!=valid[LABEL+'-post'][0]
            or before['node']!=after['node'] or before['node']!=NODE
            or before['boot_id']!=after['boot_id'] or before['pci']!=after['pci'] or len(set(before['pci']))!=8
            or before['kfd_holder_count']!=0 or after['kfd_holder_count']!=0
            or after['owned_processes_remaining']!=[] or not after['master_port_available']):
        raise ValueError('Original runtime and resource cleanup failed')
    return valid


def fingerprints(record, field, key):
    rows = record[field]
    if len(rows)!=20 or len({r['ordinal'] for r in rows})!=20:
        raise ValueError('Incomplete original twenty microbatches')
    if field=='inputs' and [r['global_step'] for r in rows]!=[step for step in range(20,25) for _ in range(4)]:
        raise ValueError('Original step and accumulation order differs')
    out = []
    for row in rows:
        e = row[key]
        if not e['finite'] or not re.fullmatch(r'[0-9a-f]{64}',e['SHA256']):
            raise ValueError('Invalid original tensor witness')
        out.append((row['ordinal'], e['SHA256']))
    return out


def compare_ranks():
    records = {}
    scopes = {}
    for phase, witness in (('continuous','continuous-tail'),('resumed','resume-witness')):
        path=CONTINUOUS if phase=='continuous' else ROOT/'resumed'
        assigned, assigned_pin=small(path/'assigned.json')
        records[phase]=[]
        scopes[phase]=dict(device_count=len(set(assigned['PCI_by_index'])),assigned_pin=assigned_pin,
                           step=assigned['step'],GPU_placement_verified=assigned['node']==NODE,
                           mask_counts=Counter(),maximum_query_tokens=0)
        for rank in range(8):
            row,row_pin=small(path/'train'/witness/f'rank-{rank:02d}'/'result.json')
            route,route_pin=small(path/'train'/f'math-sdpa-rank-{rank:02d}.json')
            if not (row['rank']==rank and row['world_size']==8 and row['completed_global_step']==25
                    and row['continuous_tail_complete' if phase=='continuous' else 'native_resume_window_complete']
                    and route['rank']==rank and route['phase']==phase and route['completed_step']==25
                    and route['mode']=='MATH' and route['text_layers']==36
                    and all(type(i) is int and i>0 for i in route['mask_calls'].values())
                    and sum(route['mask_calls'].values())>0):
                raise ValueError('Unqualified actual rank receipt')
            if phase=='resumed' and len(row['RNG_loads'])!=1:
                raise ValueError('Expected one native RNG restore on each rank')
            records[phase].append(dict(inputs=fingerprints(row,'inputs','full_input'),
                                       losses=fingerprints(row,'outputs','loss'),row_pin=row_pin,route_pin=route_pin))
            scopes[phase]['mask_counts'].update(route['mask_calls'])
            scopes[phase]['maximum_query_tokens']=max(scopes[phase]['maximum_query_tokens'],route['longest_query'])
    a,b=records['continuous'],records['resumed']
    paired=[dict(rank=i, matched_inputs=sum(x==y for x,y in zip(p['inputs'],q['inputs'],strict=True)),
                 matched_losses=sum(x==y for x,y in zip(p['losses'],q['losses'],strict=True)),
                 evidence={phase:{k:v for k,v in row.items() if k.endswith('_pin')} for phase,row in (('continuous',p),('resumed',q))})
            for i,(p,q) in enumerate(zip(a,b,strict=True))]
    return dict(paired=paired,scopes=scopes,
                input_matches=sum(r['matched_inputs'] for r in paired),loss_matches=sum(r['matched_losses'] for r in paired))


def main():
    intent,intent_pin=small(ROOT/'intent.json')
    if intent['request_id']!=REQUEST:
        raise ValueError('Wrong original run')
    outcome,outcome_pin=small(ROOT/'outcome.json',missing=True)
    rows=[line.split('|') for line in query('sacct','-n','-P','-j',PARENT,'--format='+COLUMNS)
          if line.split('|')[1] in (LABEL,LABEL+'-pre',LABEL+'-post')]
    live=query('squeue','-h','-s','-j',PARENT,'-o','%i')
    data=dict(request=REQUEST,original_source_publication=intent['claim_commit'],
              epoch=intent['epoch'], deadline=intent['deadline'], intent_pin=intent_pin, at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              accounting_columns=COLUMNS.split(','),account=rows,live_step_ids=[r[0] for r in rows if r[0] in live],
              original_outcome_pin=outcome_pin,ready_to_consume=False,comparison=None)
    if not outcome or not all(terminal_state(row) for row in rows) or data['live_step_ids']:
        return data
    data['ready_to_consume']=True
    if outcome['epoch']!=intent['epoch'] or outcome['claim_commit']!=intent['claim_commit']:
        raise ValueError('Original outcome identity differs')
    native,native_pin=small(ROOT/'native-outcome.json',missing=True)
    data['native_receipt']=native_pin
    data['original_keeper_passed']=outcome['keeper_passed']
    data['GPU_step_requested']=outcome['GPU_step_requested']
    data['post_clean']=bool(outcome['after'] and outcome['after']['owned_processes_remaining']==[]
                            and outcome['after']['kfd_holder_count']==0 and outcome['after']['master_port_available'])
    for where,info in (('controller',outcome),('native',native)):
        if info and info.get('error'):
            detail=info['error']
            data[where+'_failure']=dict(type=detail.get('type'),stage=detail.get('stage'),
                                        detail_SHA256=hashlib.sha256(detail.get('message','').encode()).hexdigest(),
                                        known_context=next((x for x in ('Native checkpoint-25 is incomplete',
                                           'Native checkpoint-20 is incomplete','Original native rank processes did not complete',
                                           'Incomplete continuous or recovery record','Source path missing','Original scope epoch')
                                          if x in detail.get('message','')),None))
    if not outcome['keeper_passed']:
        return data
    valid=check_native(outcome,rows)
    comparison=compare_ranks()
    expected={'continuous':'1087656.383','resumed':valid[LABEL][0]}
    if any(r['step']!=expected[phase] or r['device_count']!=8 or not r['GPU_placement_verified']
           for phase,r in comparison['scopes'].items()):
        raise ValueError('Original device placement differs from Slurm')
    if not native['normal_native_exit']:
        raise ValueError('Missing original native success')
    result=native['result']
    if comparison['input_matches']!=sum(r['matched_inputs'] for r in result['ranks']) or comparison['loss_matches']!=sum(r['matched_losses'] for r in result['ranks']):
        raise ValueError('Rank comparison differs from native producer')
    def verify_pin(path, expected):
        size_before=path.stat().st_size
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):
                h.update(block)
        if size_before!=path.stat().st_size or size_before!=expected['bytes'] or h.hexdigest()!=expected['SHA256']:
            raise ValueError('Actual checkpoint payload differs from pinned bytes')
        return h.hexdigest()
    for filename in ('adapter_model.safetensors','optimizer.pt','scheduler.pt',*(f'rng_state_{rank}.pth' for rank in range(8))):
        a=verify_pin(CONTINUOUS/'train/checkpoint-25'/filename,result['continuous25'][filename])
        b=verify_pin(ROOT/'resumed/train/checkpoint-25'/filename,result['resumed25'][filename])
        if (a==b)!=result['checkpoint25_equal'][filename]:
            raise ValueError('Original state comparison differs')
    for filename in ('adapter_model.safetensors','optimizer.pt','scheduler.pt'):
        verify_pin(CONTINUOUS/'train/checkpoint-20'/filename,result['checkpoint20'][filename])
    comparison['checkpoint25_byte_equality']=result['checkpoint25_equal']
    comparison['candidate_checkpoint20_SHA256']=result['checkpoint20']['adapter_model.safetensors']['SHA256']
    comparison['exact_checkpoint25']=all(result['checkpoint25_equal'].values())
    data['comparison']=comparison
    return data


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,allow_nan=False))
