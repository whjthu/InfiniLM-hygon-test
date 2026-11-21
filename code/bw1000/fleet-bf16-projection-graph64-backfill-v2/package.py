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
from scope import PCI,PILOT_SHA,REQUEST,ledger,placement,validate_plan
sys.path.insert(0,str(REPO/'tools'))
from report_training80_backlog import fleet_view


LEDGER=Path('coordination/training/training80-active-node-task-map-v1.json')
ORIGINAL=Path('results/bw1000/2026-11-22/O01-Graph64-32card-original687-684-four-node-paired-measurement.json')


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
    for node in sorted(candidates-set(protected)):
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
    refs,files=source(repo)
    selected=select(snapshot,protected,requested,repo=repo)
    owners=ledger(files['ledger.json'])
    epoch=float(campaign_epoch)
    plan=dict(request=REQUEST,campaign_epoch=epoch,shared_start_epoch=epoch+60,deadline_epoch=epoch+125,
              publication=publication,source_SHA256=identity(files),
              before_snapshot_SHA256=snapshot['local_receipt']['payload_SHA256'],pilot_SHA256=PILOT_SHA,
              selected=selected,shards=placement(selected,owners))
    validate_plan(plan,owners)
    envelope=dict(plan=plan,files={name:[sha(raw),base64.b64encode(zlib.compress(raw)).decode('ascii')]
                                   for name,raw in sorted(files.items())})
    body=('PAYLOAD='+repr(envelope)+'\n').encode('ascii')+files['controller.py']
    return plan,body,[ORIGINAL.as_posix(),str(Path(__file__).resolve().relative_to(REPO).as_posix()),
                      *(p.as_posix() for p in refs.values())]
