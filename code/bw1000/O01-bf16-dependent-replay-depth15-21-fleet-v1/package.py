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
from contract import FIRST_ACTIVE, assign_cohorts
from controller import identity,need,sha,verify_source
from scope import EXCLUDED_NODES,PCI,PILOT_SHA,REQUEST,WORKER_SECONDS,ledger,placement,validate_plan
sys.path.insert(0,str(REPO/'tools'))
from report_training80_backlog import fleet_view


LEDGER=Path('coordination/training/training80-active-node-task-map-v1.json')
ORIGINAL=Path('results/bw1000/2026-11-22/O01-depth9-15-592-card-original-terminal-and-clean-release.json')
V5_TERMINAL=Path('results/bw1000/2026-11-22/fleet-graph-depth-v5-original584-terminal-584-valid.json')
V5_TERMINAL_SHA='d4075d61f47a07a8d7bfbab3e68343605eae91e6f6913e46494bef2fa4a1d8ef'
T01_PROVIDER=Path('code/bw1000/training-qwen-native-vproj-row-alignment-v1/verify_package.py')
T01_PROVIDER_SHA='f43900be14e434c040faa3b4216395c3d5adb383ee3990e743f6cc18b0fcf16d'
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
    """Select fresh whole nodes within current ledger ownership and priority holds."""
    now=time.monotonic()
    _,facts,fresh=fleet_view(snapshot,now,time.time()-now,repo/LEDGER)
    need(fresh,'new CPU-only dispatch snapshot required')
    candidates=set(facts) if requested is None else set(requested)
    need(candidates and candidates<=set(facts) and set(protected)<=set(facts),
         'requested or protected nodes differ from the 80-node ledger')
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
    need(chosen,'no freshly clear unprotected whole nodes')
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
    trial = result["paired_comparison"]
    need(sha(raw)==PILOT_SHA and result["request"]=="029be71b711f4b5c8cfd892f65c56021"
         and result["slurm_termination_verified"] and result["queue_closed_shards"] == 31
         and result["valid_cards"] == 592 and result["independent_clean_nodes"] == 74
         and result["completed_arms"] == 2960 and result["actual_busy_card_samples"] == 35520
         and trial["measured_pairs"] == trial["planned_pairs"] == 1184
         and trial["median_depth15_over_depth9"] == 1.042445488394582,
         "the independently closed 592-card depth 9/15 comparison is required")
    previous=read(repo/V5_TERMINAL)
    v5=json.loads(previous)
    post=v5['independent_post']
    need(sha(previous)==V5_TERMINAL_SHA and post['closed_shards']==31 and post['closed_cards']==584
         and post['verified_worker_counters']['valid_cards']==584 and v5['promote_graph32'] is False,
         'v5 original terminal baseline differs')
    provider=read(repo/T01_PROVIDER)
    need(sha(provider)==T01_PROVIDER_SHA and b'config["num_key_value_heads"],config["head_dim"])==(False,4096,8,128)' in provider,
         'exact T01 V projection provider or KV-head layout differs')
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
    shards=placement(selected,owners)
    plan=dict(request=REQUEST,campaign_epoch=epoch,shared_start_epoch=epoch+FIRST_ACTIVE[0],deadline_epoch=epoch+WORKER_SECONDS,
              publication=publication,source_SHA256=identity(files),
              before_snapshot_SHA256=snapshot['local_receipt']['payload_SHA256'],pilot_SHA256=PILOT_SHA,
              selected=selected,shards=shards,cohort_by_shard=assign_cohorts(shards))
    validate_plan(plan,owners)
    envelope=dict(plan=plan,files={name:[sha(raw),base64.b64encode(zlib.compress(raw)).decode('ascii')]
                                   for name,raw in sorted(files.items())})
    body=('PAYLOAD='+repr(envelope)+'\n').encode('ascii')+files['controller.py']
    return plan,body,[ORIGINAL.as_posix(),V5_TERMINAL.as_posix(),T01_PROVIDER.as_posix(),QWEN_VJP.as_posix(),QWEN_SHAPE.as_posix(),str(Path(__file__).resolve().relative_to(REPO).as_posix()),
                      *(p.as_posix() for p in refs.values())]
