"""Current holder and first-batch evidence for the new Gemma owner."""

from datetime import datetime, timezone
from pathlib import Path
import json
import re
import time

from contract import D, LABEL, NODE, PARENT, R, REQUEST, allocated, need, stable_json
from root_guard import STABLE_TERMINAL, holder_accounting


NUMBERED = re.compile(PARENT + r'\.([0-9]+)')
MINIMUM_LATEST_HOLDER = 4010
ORIGINAL = {
    '1103606.3504': ('training-gemma-usgs-r46-two-updates-native-v1-gpu','FAILED','1:0','108',NODE,
        'cpu=64,gres/dcu=8,mem=224000M,node=1','2026-11-25T15:45:39','2026-11-25T15:47:27'),
    '1103606.3505': ('training-gemma-usgs-r46-two-updates-native-v1-post','COMPLETED','0:0','0',NODE,
        'cpu=2,mem=4G,node=1','2026-11-25T15:47:30','2026-11-25T15:47:30'),
}


def step_number(value):
    match = NUMBERED.fullmatch(value)
    need(match is not None, 'R119_REAL_NUMERIC_STEP_REQUIRED')
    return int(match[1])


def interval(row, now):
    def ts(raw):
        return datetime.strptime(raw,'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    start,end = map(ts,row[7:9])
    need(row[4].isdecimal() and 0 < start <= end <= now+1 and abs(end-start-int(row[4])) <= 1,
         'R119_REAL_ACCOUNTING_INTERVAL')
    return start,end


def exact_row(rows, expected):
    need(len(expected)==9 and [r for r in rows if r[0]==expected[0]]==[expected],
         'R119_ACCOUNTING_IDENTITY_MISMATCH')


def holder_gate(proof, rows, hosts, now):
    need(proof['request_id']==REQUEST and proof['node']==NODE and proof['parent']==PARENT
         and proof['independent'] is proof['released'] is proof['same_boot_and_pci'] is True
         and proof['open_kfd_holders']==proof['scope_owned_processes_remaining']==0
         and re.fullmatch('[0-9a-f]{64}',proof['original_release_receipt_SHA256']),
         'R119_CURRENT_INDEPENDENT_RELEASE_REQUIRED')
    holder,post = proof['holder_accounting'],proof['post_accounting']
    for step, fields in ORIGINAL.items():
        exact_row(rows,[step,*fields])
    number = step_number(holder[0])
    need(number>=MINIMUM_LATEST_HOLDER and step_number(post[0])>number and holder[5]==post[5]==NODE
         and holder[2] in STABLE_TERMINAL and allocated(holder[6])==8
         and post[2:4]==['COMPLETED','0:0'] and allocated(post[6])==0,
         'R119_LATEST_EIGHT_DCU_AND_ZERO_GRES_POST')
    exact_row(rows,holder)
    exact_row(rows,post)
    _,end = interval(holder,now)
    start,_ = interval(post,now)
    need(end<=start, 'R119_LATEST_POST_FOLLOWS_HOLDER')
    seen = set()
    for row in rows:
        match = NUMBERED.fullmatch(row[0])
        if not match or int(match[1])<=number:
            continue
        if row[5] in ('','(null)','N/A','Unknown','None assigned'):
            need(allocated(row[6])==0 and row[2] in STABLE_TERMINAL, 'R119_LATER_UNKNOWN_PLACEMENT')
            continue
        if NODE not in hosts(row[5]):
            continue
        need(row[0] not in seen and allocated(row[6])==0 and row[2] in STABLE_TERMINAL,
             'R119_LATER_HOLDER_OR_LIVE_CPU_CONFLICT')
        interval(row,now)
        seen.add(row[0])


def verify_latest_r119(query,clock):
    claim = json.loads((D/'CLAIM.json').read_bytes())
    proof = R/'incoming'/(LABEL+'-authority')/(REQUEST+'.latest-release.json')
    value, identity = stable_json(proof,cap=16<<10)
    need(identity['SHA256']==claim['latest_holder_observation_SHA256'], 'R119_FRESH_RELEASE_RECEIPT_PIN')
    def hosts(spec):
        if re.fullmatch(r'f[0-9]{2}r[0-9]+n[0-9]{2}',spec):
            return {spec}
        return set(query(['scontrol','show','hostnames',spec],clock).splitlines())
    holder_gate(value,holder_accounting(query,clock),hosts,time.time())


FIELDS = {'input_ids','labels','pixel_values'}
CHECKS = {'token_dtype_int64','label_dtype_int64','token_label_shape_match','token_rank_two',
          'batch_size_one','at_most_512_tokens','pixels_floating','pixels_rank_at_least_three'}


def rank_gate_receipts(operation, gpu, request):
    result = {'request':request,'actual_gpu_step':gpu['id'] if gpu else None,'planned_ranks':8,
              'planned_inputs':64,'planned_updates':2,'metadata_receipts':[],'missing_ranks':[],'invalid_ranks':[]}
    base = Path(operation)/'continuous/train/continuous-witness'
    for rank in range(8):
        path = base/f'rank-{rank:02d}'/'native-batch-gate-failure.json'
        try:
            value,identity = stable_json(path,cap=8192)
            need(gpu and request==REQUEST and value['rank']==rank and 1<=value['microbatch']<=8
                 and value['global_step']==(value['microbatch']-1)//4, 'R119_RANK_EVIDENCE_IDENTITY')
            meta=value['logical_input_metadata']
            need(set(meta)=={'checks','shapes','dtypes'} and set(meta['checks'])==CHECKS
                 and all(type(v) is bool for v in meta['checks'].values()) and not all(meta['checks'].values())
                 and set(meta['shapes'])==set(meta['dtypes'])==FIELDS
                 and all(isinstance(shape,(tuple,list)) and len(shape)<=8
                         and all(type(dim) is int and 0<=dim<=2**31 for dim in shape)
                         for shape in meta['shapes'].values())
                 and all(re.fullmatch(r'torch\.[a-z0-9_]{1,24}',dtype) for dtype in meta['dtypes'].values()),
                 'R119_BOUNDED_METADATA_ONLY')
            result['metadata_receipts'].append({'rank':rank,'microbatch':value['microbatch'],'pin':identity,
                                                 'logical_input_metadata':meta})
        except FileNotFoundError:
            result['missing_ranks'].append(rank)
        except (OSError,ValueError,TypeError,KeyError):
            result['invalid_ranks'].append(rank)
    return result
