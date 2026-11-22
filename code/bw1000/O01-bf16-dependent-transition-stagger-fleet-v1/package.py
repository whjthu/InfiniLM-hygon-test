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
from scope import EXCLUDED_NODES,PCI,PILOT_SHA,PRIORITY_NODES,REQUEST,WORKER_SECONDS,ledger,placement,validate_plan
sys.path.insert(0,str(REPO/'tools'))
from report_training80_backlog import fleet_view


TRANSITION_EVIDENCE=Path("results/bw1000/2026-11-22/training80-fleet640-event-join-two-valid-windows.json")
TRANSITION_EVIDENCE_SHA="dd19ea627d7d76aff685e465991675665f52d6a4b15e7e8ca9a60c0465ff8eb5"

JOIN_BASELINE=Path('results/bw1000/2026-11-22/O01-dependent-d2h-chunks-original584-terminal.json')
JOIN_BASELINE_SHA='2b91874bb8e165e58ce633c6eda8d6c056df34a1c543dacfba0c5789be7bd363'

TRANSFER_BASELINE=Path('results/bw1000/2026-11-22/O01-dependent-transfer-overlap-fleet-original592-terminal.json')
TRANSFER_BASELINE_SHA='7ded16778c5faa0f29cca323f3e7e3f19894c9d7ff9dbd8771d5d87b066c922d'

LEDGER=Path('coordination/training/training80-active-node-task-map-v1.json')
ORIGINAL=Path('results/bw1000/2026-11-22/O01-dependent-sync-cadence-fleet-original592-terminal.json')
D2H_REFERENCE=Path('results/bw1000/2026-11-22/O01-projection-contention-original669-eight-card-paired-measurement.json')
D2H_REFERENCE_SHA='7741f6819cda43588c8ea11e71db925eeecbb2e9262dd91013a7e437796dce2a'
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
    for node in sorted(candidates-set(protected)-EXCLUDED_NODES-PRIORITY_NODES):
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
    observed_raw = read(repo / TRANSITION_EVIDENCE)
    observed = json.loads(observed_raw)
    need(sha(observed_raw) == TRANSITION_EVIDENCE_SHA and observed["coobserved_operator_cards"] == 592
         and observed["records"][1]["points"][2]["inside_operator_nonzero"] == 592
         and observed["records"][1]["points"][3]["inside_operator_nonzero"] == 312,
         "original synchronized changeover observation differs")
    fixed_raw = read(repo / JOIN_BASELINE)
    fixed = json.loads(fixed_raw)
    need(sha(fixed_raw) == JOIN_BASELINE_SHA and fixed["request"] == "93c5218eea4e4f1d8a246ba7583dac85"
         and fixed["device_qualified"] is True and fixed["independent_post_clean_nodes"] == 73
         and (fixed["valid_cards"], fixed["completed_arms"], fixed["actual_busy_card_samples"]) == (584, 2920, 35040)
         and fixed["paired_comparison"]["measured_pairs"] == 1168
         and fixed["paired_comparison"]["median_two_chunks_over_one_copy"] == 0.998266112045002,
         "the actual single-copy retention evidence differs")
    measured_raw = read(repo / TRANSFER_BASELINE)
    measured = json.loads(measured_raw)
    summary = measured["summary"]
    need(sha(measured_raw) == TRANSFER_BASELINE_SHA
         and measured["request"] == "a02a3ada39494c0cec0c233639025f02"
         and summary["device_qualified"] is True and measured["post"]["clean_nodes"] == 74
         and (summary["valid_cards"], summary["completed_arms"], summary["actual_busy_card_samples"]) == (592, 2960, 35520)
         and summary["paired_comparison"]["measured_pairs"] == 1184
         and summary["paired_comparison"]["median_overlap_over_serial"] == 1.0297971503688794
         and measured["workload"] == dict(transfer_bytes_per_unit=761856, graphs=2,
                                          logical_BF16_chains=36, equal_stream_fences=2),
         "the original measured whole-fleet overlap baseline differs")
    raw=read(repo/ORIGINAL)
    result=json.loads(raw)
    summary=result["summary"]
    need(sha(raw)==PILOT_SHA and result["request"]=="4eb9dca180960f7179836fc671c6bf12"
         and summary["device_qualified"] is True and result["post"]["clean_nodes"]==74
         and (summary["valid_cards"],summary["completed_arms"],summary["actual_busy_card_samples"])==(592,2960,35520)
         and summary["paired_comparison"]["median_sync2_over_sync1"]==1.0313268208226523,
         "the actual independently qualified 592-card cadence result is required")
    transfer_raw=read(repo/D2H_REFERENCE)
    transfer=json.loads(transfer_raw)
    need(sha(transfer_raw)==D2H_REFERENCE_SHA and transfer["actual_slurm_row"][:4]==[
        "1103606.669","O01-projection-hip-copy-contention-host64-v2","COMPLETED","0:0"]
        and transfer["independent_post"]["post_clean"] is True and transfer["independent_post"]["post_probe"]=="1103606.672"
        and transfer["valid_cards"]==8,
        "the original eight-card measured HIP pinned transfer path is required")
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
        need(not requested.intersection(EXCLUDED_NODES|PRIORITY_NODES), 'explicit requested nodes include an original blocked node')
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
    script=('PAYLOAD='+repr(envelope)+'\n').encode('ascii')+files['controller.py']
    need(len(script)<=262144, 'original unpacked controller envelope exceeded')
    packed=base64.b64encode(zlib.compress(script,9)).decode('ascii')
    bootstrap=("import base64, hashlib, zlib\n"
               "decoder=zlib.decompressobj()\n"
               f"script=decoder.decompress(base64.b64decode({packed!r}, validate=True),262145)\n"
               f"if len(script)!={len(script)} or not decoder.eof or decoder.unused_data or "
               f"hashlib.sha256(script).hexdigest()!={sha(script)!r}:\n"
               "    raise ValueError('original frozen controller wrapper changed')\n"
               "exec(compile(script,'<staggered-controller>','exec'))\n")
    body=bootstrap.encode('ascii')
    return plan,body,[TRANSITION_EVIDENCE.as_posix(),JOIN_BASELINE.as_posix(),TRANSFER_BASELINE.as_posix(),ORIGINAL.as_posix(),D2H_REFERENCE.as_posix(),V5_TERMINAL.as_posix(),T01_PROVIDER.as_posix(),QWEN_VJP.as_posix(),QWEN_SHAPE.as_posix(),str(Path(__file__).resolve().relative_to(REPO).as_posix()),
                      *(p.as_posix() for p in refs.values())]
