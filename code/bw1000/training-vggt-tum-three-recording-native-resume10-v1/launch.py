"""Reserve the sole T12 node and send one pinned, finite VGGT controller."""

import base64
from contextlib import closing
from hashlib import sha256
import json
import os
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

REQUEST = "adce8503c5a14bb5acfb4e851e6d337c"
PARENT, NODE = "1103606", "f13r4n09"
BASE = HERE.relative_to(REPO)
DIRECTORY = Path("local-only")/(HERE.name+"-original")
ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results"

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
with (directory/'controller.log').open('xb') as log:
    child=subprocess.Popen(['/usr/local/python3.12/bin/python3','-I','-S','-B',str(program)],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(request=p['request'],controller_pid=child.pid,body_SHA256=p['sha'])
with (directory/'launch.json').open('x') as handle:
    json.dump(result,handle,allow_nan=False)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(result,sort_keys=True),flush=True)
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


def frozen(repo, publication):
    claim_raw=(repo/BASE/"CLAIM.json").read_bytes()
    claim=json.loads(claim_raw)
    need((claim["request"],claim["parent"],claim["node"],claim["planned_updates"],claim["training_GO"])==
         (REQUEST,PARENT,NODE,15,True) and claim["training_READY"] is False,
         "published one-run training claim differs")
    source={}
    for filename,pin in claim["sources"].items():
        rel=Path(pin["path"])
        need(not rel.is_absolute() and ".." not in rel.parts,"invalid published source path")
        raw=(repo/rel).read_bytes()
        need(file_sha(raw)==pin["sha"],"frozen source differs: "+filename)
        source[filename]=raw
    payload=dict(publication=publication,request=REQUEST,files={
        "src/"+name:dict(sha=file_sha(raw),bytes=len(raw),b64=base64.b64encode(raw).decode("ascii"))
        for name,raw in source.items()})
    body=("PAYLOAD="+repr(payload)+"\n").encode("ascii")+source["run_tum_continuous_once.py"]
    params=dict(request=REQUEST,sha=file_sha(body),body=base64.b64encode(zlib.compress(body)).decode("ascii"),
                directory=ROOT+"/"+HERE.name+"-bootstrap")
    program=("PAYLOAD="+repr(params)+"\n").encode("ascii")+BOOTSTRAP.encode("ascii")
    need(len(program)<=131072,"controller transport envelope exceeded")
    return claim, file_sha(claim_raw), file_sha(body), program


def ship(publication, snapshot, transport, *, repo=REPO, check_publication=published):
    repo=repo.resolve(strict=True)
    claim,claim_sha,body_sha,program=frozen(repo,publication)
    check_publication(repo,publication,[(BASE/name).as_posix() for name in ("launch.py","CLAIM.json","SOURCE_FILES.json")]
                      +[pin["path"] for pin in claim["sources"].values()])
    now=time.monotonic()
    _,facts,fresh=fleet_view(snapshot,now,time.time()-now,repo/"coordination/training/training80-active-node-task-map-v1.json")
    receipt=snapshot["local_receipt"]
    cutoff=receipt["received_monotonic"]+SNAPSHOT_MAX_AGE_S-receipt["roundtrip_s"]
    need(fresh and now<cutoff and facts[NODE]["admission"]=="clear" and facts[NODE]["parent"]==PARENT
         and len(snapshot["nodes"][NODE]["PCIs"])==8 and snapshot["nodes"][NODE]["active_steps"]==[],
         "fresh exact one-node training admission required")
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
            packet=dict(candidate=HERE.name,case=claim["case"],kind="training",request=REQUEST,
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
                record["returncode"]=transport(program,out,error,35).returncode
        except Exception as error:
            record["transport_error"]=type(error).__name__
        save(directory/"transport.json",record)
    return record
