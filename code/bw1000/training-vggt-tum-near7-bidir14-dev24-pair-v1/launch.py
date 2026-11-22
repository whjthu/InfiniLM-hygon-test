"""Reserve the sole T12 node and send one pinned, finite VGGT controller."""

import base64
from contextlib import closing
from hashlib import sha256
import subprocess
import json
import os
import re
from pathlib import Path
import time
import zlib

import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0,str(REPO/"tools"))
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, fleet_view, plan, queue_holds
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest

REQUEST = "d282f24a26e14b35b3e50bf02adbf9ff"
PARENT, NODE = "1103606", "f13r4n09"
LABEL = HERE.name
PACKAGE = HERE
BASE = HERE.relative_to(REPO)
DIRECTORY = Path("local-only")/(HERE.name+"-original")
ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results"
NEAR7_PUBLIC = "results/bw1000/2026-11-22/training-vggt-near7-bidir14-original1078-native25-terminal.json"
NEAR7_PUBLIC_SHA = "3e3a0f55e2f4ac7f375728e3e84c8f673ce1b5c7e64734491433d14d121bb7be"

AUXILIARY = ('audit_terminal.py', 'read_vggt_dev24_roundoff_audit.py', 'dev24_decision.py')
ORIGINAL_INDEX = 'coordination/training/vggt-tum-cross-camera-controller-files-v1.json'
ORIGINAL_INDEX_SHA = 'f224f502dc9b5755182f458591e47cf5214f7484dc0c0480d4028d5dade0a910'
CONTROLLER = 'tools/control_vggt_tum_cross_camera_dev.py'
STAGE_SHA = 'd221478fe51e14ccdf8ca858632c60aee67993e693c58f29b2c9357d373dc903'

BOOTSTRAP = r"""
import base64,hashlib,json,os,pathlib,subprocess,zlib
p=PAYLOAD
body=zlib.decompress(base64.b64decode(p['body'],validate=True))
if hashlib.sha256(body).hexdigest()!=p['sha']:
    raise ValueError('original controller body changed')
os.umask(0o077)
directory=pathlib.Path(p['directory'])
directory.mkdir(mode=0o700)
program=directory/'controller.py'
with program.open('xb') as handle:
    handle.write(body)
    handle.flush()
    os.fsync(handle.fileno())
# The root's authenticated held transport remains attached; never detach a controller.
os.execv('/usr/local/python3.12/bin/python3',
         ['/usr/local/python3.12/bin/python3','-I','-S','-B',str(program)])
"""


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def file_sha(raw):
    return sha256(raw).hexdigest()


def save(path, value):
    with path.open("x",encoding="utf8") as out:
        json.dump(value,out,sort_keys=True,allow_nan=False)
        out.flush()
        os.fsync(out.fileno())


def read_pinned(publication, relative, wanted=None):
    raw = (REPO / relative).read_bytes()
    if wanted is not None and sha256(raw).hexdigest() != wanted:
        raise ValueError(f'original source changed: {relative}')
    command = ['git', '--no-optional-locks', '-c', 'safe.directory=' + REPO.as_posix(),
               '-C', str(REPO), 'cat-file', 'blob', publication + ':' + relative]
    if subprocess.check_output(command, timeout=10) != raw:
        raise ValueError(f'publication source changed: {relative}')
    return raw


def replace_unique(raw, changes):
    for old, new in changes:
        need(raw.count(old) == 1, "frozen scoring source boundary differs")
        raw = raw.replace(old,new)
    return raw


def controller_sources(publication, read=read_pinned):
    original = json.loads(read(publication, ORIGINAL_INDEX, ORIGINAL_INDEX_SHA))
    sources = {name: read(publication, name, pin) for name, pin in original.items()}
    source = sources[CONTROLLER]
    for old, new in (
        (b"PARENT, NODE = '1087656', 'f11r4n19'", f"PARENT, NODE = '{PARENT}', '{NODE}'".encode()),
        (b"LABEL = 'training-vggt-tum-cross-camera-dev24-v1'", f"LABEL = '{LABEL}'".encode()),
    ):
        if source.count(old) != 1:
            raise ValueError('original controller binding changed')
        source = source.replace(old, new)
    snapshot_fields = (
        (b'def clean_snapshot(root):', b'def clean_snapshot(root, phase):'),
        (b'    print(json.dumps(snap), flush=True)\n', b'    write_json(root / (phase + "-device.json"), snap)\n'),
        (b"stages['pre'] = run_step(root, 'pre', command, deadline)", b"stages['pre'] = run_step(root, 'pre', [*command, '--snapshot-phase', 'pre'], deadline)"),
        (b"stages['post'] = run_step(root, 'post', command, deadline)", b"stages['post'] = run_step(root, 'post', [*command, '--snapshot-phase', 'post'], deadline)"),
        (b"read_object(root / 'pre.log')", b"read_object(root / 'pre-device.json')"),
        (b"read_object(root / 'post.log')", b"read_object(root / 'post-device.json')"),
        (b"    for flag in ('stage-sha', 'request', 'publication', 'manifest-sha'):", b"    parser.add_argument('--snapshot-phase', choices=('pre', 'post'))\n    for flag in ('stage-sha', 'request', 'publication', 'manifest-sha'):"),
        (b"    if args.worker:\n        return {'snapshot': clean_snapshot, 'native': native}[args.worker](args.root)", b"    if args.worker == 'snapshot':\n        require(args.snapshot_phase is not None, 'snapshot phase is required')\n        return clean_snapshot(args.root, args.snapshot_phase)\n    require(args.snapshot_phase is None, 'snapshot phase is exclusive to its worker')\n    if args.worker == 'native':\n        return native(args.root)"),
    )
    for old, new in snapshot_fields:
        if source.count(old) != 1:
            raise ValueError('original snapshot boundary differs')
        source = source.replace(old, new)
    source = replace_unique(source, (
        (b"                '--head-authority-sha', intent['head_authority_SHA256'], '--alternate', '/alternate',\n"
         b"                '--lowlr', '/lowlr', '--upstream', '/vggt-source', '--weights', '/weights/model.pt',",
         b"                '--head-authority-sha', intent['head_authority_SHA256'], '--near7', '/near7',\n"
         b"                '--upstream', '/vggt-source', '--weights', '/weights/model.pt',"),
        (b"             (RESULTS / SOURCES['alternate25'][0], '/alternate:ro'), (RESULTS / SOURCES['lowlr25'][0], '/lowlr:ro'),",
         b"             (RESULTS / 'training-vggt-tum-near7-bidirectional14-native25-v1', '/near7:ro'),"),
    ))
    compile(source, CONTROLLER, 'exec')
    sources[CONTROLLER] = source

    scorer_path = 'tools/score_vggt_tum_cross_camera_dev.py'
    scorer = sources[scorer_path]
    substitutions = (
        (b'import numpy as np', b'import importlib.metadata\nimport numpy as np'),
        (b'torch.__version__ != TORCH_VERSION', b'importlib.metadata.version("torch") != TORCH_VERSION\n            or str(torch.version.__version__) != "2.10.0"\n            or str(torch.__version__) not in ("2.10.0", TORCH_VERSION)'),
    )
    for old, new in substitutions:
        if scorer.count(old) != 1:
            raise ValueError('original scoreboard provider check changed')
        scorer = scorer.replace(old, new)
    scorer = replace_unique(scorer, (
        (b"from bind_vggt_tum_cross_camera_heads import ARMS, HEAD, is_sha", b'from bind_vggt_tum_cross_camera_heads import HEAD, is_sha\nARMS = {"near7_bidir25": ("1103606.1078", "3bd3a976cc5348148e982f7e3d8fe0b6", "1103606.1081", "ec6b63f6166dfc5fa67af17dffeced85e210e37407df603bfe2fb55d242d9959")}'),
        (b'    roots = {"alternate25": args.alternate, "lowlr25": args.lowlr}', b'    roots = {"near7_bidir25": args.near7}'),
        (b'"head-authority", "alternate", "lowlr", "upstream"', b'"head-authority", "near7", "upstream"'),
        (b'expected the two independently verified training heads', b'expected the original verified bidirectional near7 head'),
    ))
    compile(scorer, scorer_path, 'exec')
    sources[scorer_path] = scorer
    stage_path = 'tools/stage_vggt_tum_cross_camera_dev.py'
    original_staged = b'training-vggt-tum-cross-camera-dev24-v1-staged'
    if sources[stage_path].count(original_staged) != 1:
        raise ValueError('original scoring stage identity changed')
    sources[stage_path] = sources[stage_path].replace(
        original_staged, LABEL.encode())
    index = {name: sha256(data).hexdigest() for name, data in sources.items()}
    sources[ORIGINAL_INDEX] = (json.dumps(index, sort_keys=True, allow_nan=False) + '\n').encode()
    return sources


def frozen(repo, publication, *, epoch, deadline, read=read_pinned):
    claim_raw=(repo/BASE/"CLAIM.json").read_bytes()
    claim=json.loads(claim_raw)
    need(re.fullmatch(r"[0-9a-f]{32}",REQUEST) and claim["label"] == LABEL
         and claim["arms"] == ["base","near7_bidir25"] and claim["planned_windows"] == 24
         and claim["admitted_windows"] == 22 and claim["expected_predictions"] == 44
         and claim["original_missing_windows"] == 2, "scope or failure denominator differs")
    need((claim["request"],claim["parent"],claim["node"],claim["GPU_GO"],claim["optimizer_updates"])==
         (REQUEST,PARENT,NODE,True,0) and claim["training_READY"] is False,
         "root operative development scoring claim differs")
    sources=controller_sources(publication,read)
    index_sha=file_sha(sources[ORIGINAL_INDEX])
    need(index_sha==claim["remapped_source_index_SHA256"],"derived scoring source index differs")
    receiver=read(publication,(BASE/"receive.py").as_posix())
    terminal=read(publication,NEAR7_PUBLIC,NEAR7_PUBLIC_SHA)
    context=dict(request=REQUEST,publication=publication,label=LABEL,stage_SHA256=STAGE_SHA,
                 source_index_SHA256=index_sha,claim_SHA256=file_sha(claim_raw),epoch=epoch,deadline=deadline)
    binding=read(publication,(BASE/"HEAD_BINDING.json").as_posix(),claim["head_binding_SHA256"])
    auxiliary={name:read(publication,(BASE/name).as_posix(),claim["auxiliary_SHA256"][name]) for name in AUXILIARY}
    context.update(head_binding_SHA256=file_sha(binding),pair_reader_SHA256=file_sha(sources["tools/audit_vggt_tum_cross_camera_dev.py"]),
                   auxiliary_SHA256=claim["auxiliary_SHA256"])
    body=("INPUT="+repr(dict(context=context,sources=sources,near7_terminal=terminal,head_binding=binding,auxiliary=auxiliary))+"\n").encode("ascii")+receiver
    params=dict(request=REQUEST,sha=file_sha(body),body=base64.b64encode(zlib.compress(body)).decode("ascii"),
                directory=ROOT+"/"+HERE.name+"-bootstrap")
    program=("PAYLOAD="+repr(params)+"\n").encode("ascii")+BOOTSTRAP.encode("ascii")
    need(len(program)<=131072,"controller transfer envelope exceeded")
    return claim,file_sha(claim_raw),file_sha(body),program


def ship(publication, snapshot, transport, *, repo=REPO, check_publication=published):
    epoch=time.time()
    mono_end=time.monotonic()+1800
    repo=repo.resolve(strict=True)
    claim,claim_sha,body_sha,program=frozen(repo,publication,epoch=epoch,deadline=epoch+1800)
    check_publication(repo,publication,[(BASE/name).as_posix() for name in ("launch.py","CLAIM.json","SOURCE_FILES.json","receive.py","HEAD_BINDING.json",*AUXILIARY)]
                      + [NEAR7_PUBLIC,ORIGINAL_INDEX])
    now=time.monotonic()
    _,facts,fresh=fleet_view(snapshot,now,time.time()-now,repo/"coordination/training/training80-active-node-task-map-v1.json")
    receipt=snapshot["local_receipt"]
    cutoff=receipt["received_monotonic"]+SNAPSHOT_MAX_AGE_S-receipt["roundtrip_s"]
    need(fresh and now<cutoff and facts[NODE]["admission"]=="clear" and facts[NODE]["parent"]==PARENT
         and len(snapshot["nodes"][NODE]["PCIs"])==8 and snapshot["nodes"][NODE]["active_steps"]==[],
         "fresh exact one-node validation admission required")
    directory=repo/DIRECTORY
    queue=Queue(repo/QUEUE_DB.relative_to(REPO))
    with closing(queue.db):
        db=queue.db
        db.execute("BEGIN IMMEDIATE")
        try:
            board=plan(snapshot,json.loads((repo/BACKLOG.relative_to(REPO)).read_bytes()),repo=repo,holds=queue_holds(db))
            protected=set(board["queue_held_nodes"])
            protected.update(node for group in board["original_launch_holds"].values() for node in group["nodes"])
            protected.update(node for row in board["rows"] if row["dispatch_state"]=="proposed_for_root"
                             and row["kind"]=="training" for node in row["placement"])
            need(NODE not in protected and not directory.exists() and not directory.is_symlink()
                 and time.monotonic()<cutoff,"original hold, higher-priority work, expired receipt or duplicate")
            packet=dict(candidate=HERE.name,case=claim["case"],kind="validation",request=REQUEST,
                        placement_by_parent={PARENT:{NODE:1}},publication_commit=publication,workload_SHA256=claim_sha,
                        snapshot_payload_SHA256=receipt["payload_SHA256"],snapshot_raw_SHA256=receipt["raw_response_SHA256"],
                        snapshot_request=receipt["request_id"],expires_local_monotonic=cutoff,
                        launcher=(BASE/"launch.py").as_posix(),invocation={"kind":"pinned_finite_code"},replay_allowed=False)
            packet["handoff_SHA256"]=digest(packet)
            run=digest(dict(stage=DIRECT_STAGE,request=REQUEST))
            intent=(DIRECTORY/"intent.json").as_posix()
            reserved=reserve_handoff(queue,run,packet,intent)
            directory.mkdir(mode=0o700)
            save(directory/"intent.json",dict(request=REQUEST,publication=publication,body_SHA256=body_sha,
                                             run=run,handle=reserved["handle"],packet=packet,replay_allowed=False))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        record=dict(request=REQUEST,body_SHA256=body_sha,transport_attempted=False,remote_outcome="NOT_ATTEMPTED")
        try:
            with (directory/"stdout.private").open("xb") as out,(directory/"stderr.private").open("xb") as error:
                need(time.monotonic()<cutoff,"expired original admission")
                queue.submission(run,reserved["handle"],unknown=True)
                need(time.monotonic()<cutoff,"expired original admission before transport")
                record.update(transport_attempted=True,remote_outcome="UNKNOWN")
                record["returncode"]=transport(program,out,error,max(1,mono_end-time.monotonic())).returncode
        except Exception as error:
            record["transport_error"]=type(error).__name__
        save(directory/"transport.json",record)
    return record
