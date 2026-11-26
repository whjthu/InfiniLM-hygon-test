"""Require eight durable same-activation votes before a timed real rank barrier."""

from datetime import timedelta
from hashlib import sha256
import json
import os
import re
import stat
import time

from rendezvous import activation, private_directory, write_once, DIAGNOSTIC, WAIT
from swift_sampler_gate import ARROW_SHA,CPU_READ,ident,need


VOTE_SCHEMA="GEMMA_R45_REAL_SWIFT_SAMPLER_VOTE_V1"
UNANIMOUS_SCHEMA="GEMMA_R45_REAL_SWIFT_SAMPLER_UNANIMOUS_V1"


def read_vote(root,rank,active):
    path=root/f"rank-{rank:02d}"/"sampler-votes"
    vote=path/DIAGNOSTIC
    try:
        private_directory(path.parent,True)
        private_directory(path,True)
        b=vote.lstat()
        need(stat.S_ISREG(b.st_mode) and b.st_uid==os.getuid() and stat.S_IMODE(b.st_mode)==0o600
             and 0<b.st_size<=32768,"G45_BOUNDED_PRIVATE_VOTE")
        keys=lambda m:(m.st_dev,m.st_ino,m.st_size,m.st_mtime_ns,m.st_ctime_ns)
        fd=os.open(vote,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,"rb") as stream:
            opened=os.fstat(stream.fileno())
            content=stream.read(32769)
            after=os.fstat(stream.fileno())
        directory=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        need(keys(b)==keys(opened)==keys(after)==keys(vote.lstat()) and len(content)==b.st_size,
             "G45_STABLE_PRIVATE_VOTE")
        payload=json.loads(content)
        identity=payload["activation"]
        need(payload["schema"]==VOTE_SCHEMA and type(payload["rank"]) is int and payload["rank"]==rank
             and identity["activation_SHA256"]==active["activation_SHA256"] and identity["step"]==active["step"]
             and all(type(identity[k]) is int and identity[k]>0 for k in ("PID","startticks"))
             and type(payload["optimizer_updates"]) is int and payload["optimizer_updates"]==0,
             "G45_SAME_ACTIVATION_RANK_VOTE")
    except FileNotFoundError:
        return "MISSING",None,None
    except (OSError,ValueError,TypeError,KeyError):
        return "INVALID",None,None
    return "RECEIVED",payload,sha256(content).hexdigest()


def wait_votes(common,active,deadline,*,read=read_vote,wall=time.time,mono=time.monotonic,sleep=time.sleep):
    private_directory(common,False)
    stop=mono()+max(0,min(25,deadline-wall()))
    rows={}
    while True:
        for rank in range(8):
            rows[rank]=read(common,rank,active)
        if all(row[0]=="RECEIVED" for row in rows.values()) or min(stop-mono(),deadline-wall())<=0:
            return rows
        sleep(max(0,min(.2,stop-mono(),deadline-wall())))


def unanimous(rows):
    state={"missing":[rank for rank,row in rows.items() if row[0]=="MISSING"],
           "invalid":[rank for rank,row in rows.items() if row[0]=="INVALID"],
           "failed":[],"votes":{str(rank):v[2] for rank,v in rows.items() if v[0]=="RECEIVED"},
           "planned_ranks":8,"planned_rows":800,"planned_first_slots":64,
           "all_rank_votes_valid":False}
    votes={rank:row[1] for rank,row in rows.items() if row[0]=="RECEIVED"}
    state["failed"]=[rank for rank,doc in votes.items() if doc.get("passed") is not True]
    if set(votes)!=set(range(8)) or state["failed"]:
        return state
    hashes=("expected_all_rank_logical_SHA256","expected_all_rank_first8_SHA256")
    def valid_arrays(ev):
        return all(type(ev.get(k)) is list and len(ev[k])==8
                   and all(type(x) is str and re.fullmatch(r"[0-9a-f]{64}",x) for x in ev[k]) for k in hashes)
    a=votes[0].get("audit")
    if type(a) is not dict or not valid_arrays(a):
        state["failed"].append(0)
        return state
    actors=[doc.get("activation",{}).get("PID") for doc in votes.values()]
    for rank,doc in votes.items():
        ev=doc.get("audit")
        if (type(ev) is not dict or not valid_arrays(ev) or len(set(actors))!=8
                or doc.get("RNG_unchanged") is not True or doc.get("live_sampler_iterated") is not False
                or ev.get("arrow_rows")!=800 or ev.get("arrow_SHA256")!=ARROW_SHA or ev.get("logical_slots")!=100
                or ev.get("cpu_shard_reference_SHA256")!=CPU_READ
                or ev["expected_all_rank_logical_SHA256"]!=a["expected_all_rank_logical_SHA256"]
                or ev["expected_all_rank_first8_SHA256"]!=a["expected_all_rank_first8_SHA256"]
                or ev.get("logical_rank_SHA256")!=a["expected_all_rank_logical_SHA256"][rank]
                or ev.get("rank_first8_SHA256")!=a["expected_all_rank_first8_SHA256"][rank]):
            state["failed"].append(rank)
    state["all_rank_votes_valid"]=not state["failed"]
    return state


def vote_and_wait(rank_root,rank,epoch,deadline,torch,local,unchanged):
    rank_dir=rank_root/"sampler-votes"
    rank_dir.mkdir(mode=0o700)
    active=activation(epoch,deadline)
    record={"schema":VOTE_SCHEMA,"activation":active,"rank":rank,"passed":local["passed"] and unchanged,
            "RNG_unchanged":unchanged,"live_sampler_iterated":False,"optimizer_updates":0,
            "failed_predicate":local.get("failed_predicate"),"error_class":local.get("error_class"),"audit":local["audit"]}
    write_once(rank_dir,DIAGNOSTIC,record)
    rows=wait_votes(rank_root.parent,active,deadline)
    verdict=unanimous(rows)
    verdict.update(schema=UNANIMOUS_SCHEMA,activation_SHA256=active["activation_SHA256"],step=active["step"],rank=rank,
                   all_rank_barrier_passed=False)
    try:
        need(verdict["all_rank_votes_valid"],"G45_EIGHT_RANK_SAME_ACTIVATION_CONSENSUS")
        dist=torch.distributed
        seconds=max(0,min(15,deadline-time.time()))
        need(seconds>0 and dist.is_initialized() and dist.get_world_size()==8 and dist.get_rank()==rank,
             "G45_ACTUAL_BOUNDED_EIGHT_RANK_BARRIER")
        work=dist.barrier(async_op=True)
        need(work.wait(timeout=timedelta(seconds=seconds)) is True,"G45_ALL_RANK_RELEASE_BARRIER")
        verdict["all_rank_barrier_passed"]=True
    finally:
        write_once(rank_dir,WAIT,verdict)
    return verdict
