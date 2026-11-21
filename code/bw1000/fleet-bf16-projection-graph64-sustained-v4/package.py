"""Read fresh admission and assemble one finite campaign. No reservation or transport here."""

import base64
import json
from pathlib import Path
import re
import sys
import time
import zlib


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE))
from controller import identity,need,sha,verify_source
from scope import EXCLUDED_NODES,PCI,PILOT_SHA,REQUEST,WORKER_SECONDS,ledger,placement,validate_plan
sys.path.insert(0,str(REPO/'tools'))
from report_training80_backlog import fleet_view


LEDGER=Path('coordination/training/training80-active-node-task-map-v1.json')
ORIGINAL=Path('results/bw1000/2026-11-22/O01-Graph64-32card-original687-684-four-node-paired-measurement.json')

V3_TERMINAL=Path('results/bw1000/2026-11-22/fleet-graph64-v3-original48-fixed-setup-terminal-and-clear.json')
V3_TERMINAL_SHA='90f0a88349e51a6cb83441ceebc9683ae25a6a1f2bfb73171eed2a885609ba0c'
QWEN_VJP=Path('results/bw1000/2026-11-20/training-qwen-original348-native-row-vjp-terminal.json')
QWEN_VJP_SHA='35948173ea9ffd66ded83e34fae56d9922f17da97e33c1c8d8b543fda4ebbac8'
QWEN_SHAPE=Path('coordination/tasks/training-qwen-native-vproj-row-alignment-v1.json')
QWEN_SHAPE_SHA='98d3348b02d9eaf117e4bb193ee7aedf1d3bcf523aa2835559be2a245c870feb'


def read(path):
    need(path.resolve(strict=True)==path and path.is_file(),'source path is not its original canonical path')
    with path.open('rb') as stream:
        raw=stream.read(32769)
    need(0<len(raw)<=32768,'source byte envelope exceeded')
    return raw


def select(snapshot,protected,requested=None,*,repo=REPO):
    """Select only wholly clear nodes; an omitted requested list uses all verified ledger nodes."""
    now=time.monotonic()
    _,facts,fresh=fleet_view(snapshot,now,time.time()-now,repo/LEDGER)
    need(fresh,'new CPU-only dispatch snapshot required')
    candidates=set(facts) if requested is None else set(requested)
    need(candidates and candidates<=set(facts) and set(protected)<=set(facts),
         'requested or protected nodes are not in the fixed ledger')
    chosen={}
    for node in sorted(candidates-set(protected)-EXCLUDED_NODES):
        row,probe=facts[node],snapshot['nodes'][node]
        before=probe.get('device_probe_step')
        pci=probe.get('PCIs',[])
        if (row['admission']=='clear' and probe['active_steps']==[] and probe.get('probe_error') is None
                and len(pci)==len(set(pci))==8 and all(PCI.fullmatch(device) for device in pci)
                and isinstance(before,str) and re.fullmatch(row['parent']+r'\.[0-9]+',before)
                and all(type(probe.get(k)) is int and probe[k]==0
                        for k in ('kfd_process_count','own_device_fd_process_count'))):
            chosen[node]=dict(parent=row['parent'],zone=row['zone'],PCI=sorted(pci),before_CPU_step=before)
    need(bool(chosen),'no freshly clear unprotected whole node')
    return chosen


def source(repo):
    index=HERE.relative_to(REPO)/'SOURCE_FILES.json'
    raw=read(repo/index)
    refs={n:Path(v['path']) for n,v in json.loads(raw)['files'].items()}
    need(all(not p.is_absolute() and '..' not in p.parts for p in refs.values()),'invalid source relative path')
    files={n:read(repo/p) for n,p in refs.items()}
    files['SOURCE_FILES.json']=raw
    refs['SOURCE_FILES.json']=index
    verify_source(files)
    return refs,files


def prepare(publication,campaign_epoch,snapshot,protected,requested=None,*,repo=REPO):
    """Return a plan, finite-code bytes and publication members after root supplies protected nodes."""
    raw=read(repo/ORIGINAL)
    result=json.loads(raw)
    need(sha(raw)==PILOT_SHA and result['status']=='TWO_ACTUAL_SHARDS_PASS_AND_INDEPENDENT_POST_CLEAN'
         and set(row[0] for row in result['slurm_rows'])=={'1087656.687','1103606.684'}
         and all(row[2:4]==['COMPLETED','0:0'] for row in result['slurm_rows'])
         and (result['qualified_cards'],result['completed_arms'],result['actual_busy_card_samples'])==(32,128,512),
         'the independent 32-card result does not qualify the expansion')
    previous=read(repo/V3_TERMINAL)
    v3=json.loads(previous)
    need(sha(previous)==V3_TERMINAL_SHA and v3['verified']['closed_shards']==3 and
         v3['verified']['verified_worker_counters']==dict(valid_cards=48,completed_arms=192,actual_busy_card_samples=768)
         and all(row['queue_state']=='CLOSED' and row['Slurm_state']=='COMPLETED' for row in v3['actual_steps']),
         'v3 worker and post baseline differs')
    shape_raw=read(repo/QWEN_SHAPE)
    vjp_raw=read(repo/QWEN_VJP)
    need(sha(shape_raw)==QWEN_SHAPE_SHA and '186 rows' in json.loads(shape_raw)['controlled_difference']
         and sha(vjp_raw)==QWEN_VJP_SHA and json.loads(vjp_raw)['vjp_site']=='decoder_06.v_proj',
         'actual T01 geometry or negative VJP baseline differs')
    refs,files=source(repo)
    if requested is not None:
        requested=set(requested)
        need(not requested.intersection(EXCLUDED_NODES), 'explicit requested nodes include an original blocked node')
    selected=select(snapshot,protected,requested,repo=repo)
    if requested is not None:
        need(set(selected)==requested, 'one or more explicitly requested nodes lacks clearance')
    owners=ledger(files['ledger.json'])
    epoch=float(campaign_epoch)
    plan=dict(request=REQUEST,campaign_epoch=epoch,shared_start_epoch=epoch+60,deadline_epoch=epoch+WORKER_SECONDS,
              publication=publication,source_SHA256=identity(files),
              before_snapshot_SHA256=snapshot['local_receipt']['payload_SHA256'],pilot_SHA256=PILOT_SHA,
              selected=selected,shards=placement(selected,owners))
    validate_plan(plan,owners)
    envelope=dict(plan=plan,files={name:[sha(raw),base64.b64encode(zlib.compress(raw)).decode('ascii')]
                                   for name,raw in sorted(files.items())})
    body=('PAYLOAD='+repr(envelope)+'\n').encode('ascii')+files['controller.py']
    return plan,body,[ORIGINAL.as_posix(),V3_TERMINAL.as_posix(),QWEN_VJP.as_posix(),QWEN_SHAPE.as_posix(),str(Path(__file__).resolve().relative_to(REPO).as_posix()),
                      *(p.as_posix() for p in refs.values())]
