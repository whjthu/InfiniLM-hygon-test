"""Prepare or verify the InternVL 17-member, read-only, per-file bind view."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


LABEL = "training-internvl-modelroot17-assemble-CPU-v3"
REQUEST = "07e9bbe95eed4da59db4e2f3ace7ae47"
HERE = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ORIGINAL_361 = ROOT/"training-internvl-existing14-reuse-CPU-v1/audit/outcome.json"
OUTCOME_SHA = "7dc5c130049ad7779c7d41425219115d7f42402601150629fdd426311f740411"
OUTCOME_BYTES = 8887
ASSETS_SHA = "14ac17c0d9fd7aaba7e23f24b6db5bed63bd60a84e02b747816c762398eacbee"
PUBLISHED_03_CHECKPOINT_SHA = "ff080a6410711934646bd28846b1fc669ad21a4c2681ed2f88e4fcbd92fc5c97"
OWNER = 154459
MOUNT_ROOT = PurePosixPath("/internvl-model")
STAT_FIELDS = ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_uid","st_mode","st_nlink")
HEX = re.compile(r"[0-9a-f]{64}")
TOKEN = re.compile(r"[a-zA-Z0-9._-]+")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def serialized(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def stamp(info):
    return {field:getattr(info,field) for field in STAT_FIELDS}


def fd_info(path, *, cap=None, owner=OWNER):
    """Open a regular file, compare FD and pathname stamps and optionally read metadata."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            "Expected canonical, non-symlink source")
    fd = os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
    with os.fdopen(fd,"rb",buffering=0) as stream:
        before = stamp(os.fstat(stream.fileno()))
        require(stat.S_ISREG(before["st_mode"]) and before["st_uid"] == owner, "Original file type or UID differs")
        payload = None
        if cap is not None:
            require(0 < before["st_size"] <= cap, "Original metadata cap exceeded")
            payload = stream.read(before["st_size"]+1)
            require(len(payload) == before["st_size"], "Original metadata extent changed")
        require(before == stamp(os.fstat(stream.fileno())) == stamp(path.lstat()),
                "Original FD/path stamp changed")
    return before,payload


def document(path, wanted, maximum, *, extent=None, owner=OWNER):
    seen,raw = fd_info(path,cap=maximum,owner=owner)
    require(HEX.fullmatch(wanted) and digest(raw)==wanted and (extent is None or len(raw)==extent),
            "Frozen metadata bytes or SHA differ")
    return json.loads(raw),seen


def old_rows(result, expected):
    require(result["step"]=="1087656.361" and result["passed"] is True
            and result["completed_hash_files"]==14 and result["actual_hash_bytes"]==4922323586
            and result["owner"]["host"].split(".",1)[0]=="a14r3n04","Original .361 whole-file qualification differs")
    rows = {}
    for row in result["verified_members"]:
        leaf = Path(row["name"]).name
        bound = expected[leaf]
        require(leaf not in rows and (row["name"],row["bytes"],row["SHA256"]) ==
                (bound["name"],bound["bytes"],bound["SHA256"]) and row["whole_file_read_once"] is True
                and row["observer_host"] == result["owner"]["host"]
                and row["observer_boot_id"] == result["owner"]["boot_id"]
                and set(row["current_target_stat"]) == set(STAT_FIELDS),
                "Original .361 per-file identity differs")
        rows[leaf] = row["current_target_stat"]
    require(set(rows)==set(expected),"Expected all 14 originally qualified members")
    return rows


def verify_checkpoint(state, spec, final):
    require(state["spec"] == spec and state["sealed"] is True and state["confirmed_bytes"] == spec["bytes"],
            "Original public shard missing exact whole seal")
    cursor = 0
    for row in state["segments"]:
        span = min(spec["segment_bytes"],spec["bytes"]-cursor)
        require(span>0 and row["offset"]==cursor and row["bytes"]==span and HEX.fullmatch(row["SHA256"]),
                "Original durable segment denominator differs")
        cursor += span
    require(cursor==spec["bytes"]==final["st_size"] and state["file"]==[final["st_dev"],final["st_ino"]]
            and state["sealed_stat"]==[final["st_mtime_ns"],final["st_ctime_ns"]],
            "Current regular inode differs from the exact seal")


def source_receipts(assets, authority, *, owner=OWNER, pin03=PUBLISHED_03_CHECKPOINT_SHA):
    require(authority["schema"]=="INTERNVL_PUBLIC_INDEPENDENT_THREE_SEALS_V1" and authority["request"]==REQUEST
            and set(authority["seals"])==set(assets["public_shards"])=={"2","3","4"},
            "Three individually observed public seals are required")
    rows = {}
    for key, entry in sorted(assets["public_shards"].items()):
        spec = entry["spec"]
        proof = authority["seals"][key]
        require(proof["independent_sealed_inode_verified"] is True
                and type(proof["receiver_rc"]) is int and proof["receiver_rc"]==0
                and HEX.fullmatch(proof["terminal_receipt_SHA256"])
                and proof["source"]==entry["source"] and proof["checkpoint"]==entry["checkpoint"]
                and proof["file_id"]==spec["file_id"] and proof["whole_SHA256"]==spec["SHA256"]
                and type(proof["checkpoint_bytes"]) is int and 0 < proof["checkpoint_bytes"] <= 131072
                and (key!="3" or proof["checkpoint_SHA256"]==pin03),
                "Independent receiver and checkpoint identity differs")
        checkpoint, checkpoint_stamp = document(entry["checkpoint"],proof["checkpoint_SHA256"],131072,
                                                 extent=proof["checkpoint_bytes"],owner=owner)
        final,_ = fd_info(entry["source"],owner=owner)
        require(final==proof["final_stat"],"Independent final regular stamp differs")
        verify_checkpoint(checkpoint,spec,final)
        rows[spec["filename"]] = dict(source=entry["source"],bytes=spec["bytes"],SHA256=spec["SHA256"],
            stamp=final, checkpoint=entry["checkpoint"],checkpoint_stamp=checkpoint_stamp,
            checkpoint_SHA256=proof["checkpoint_SHA256"],terminal_receipt_SHA256=proof["terminal_receipt_SHA256"])
    return rows


def build_plan(assets, authority, original, outcome, *, owner=OWNER, pin03=PUBLISHED_03_CHECKPOINT_SHA):
    sealed = source_receipts(assets,authority,owner=owner,pin03=pin03)
    old = old_rows(outcome,assets["old14"])
    rows,mismatched = {},[]
    for leaf,entry in sorted(assets["old14"].items()):
        seen,_ = fd_info(entry["source"],owner=owner)
        if seen!=old[leaf]:
            mismatched.append(dict(name=leaf,source=entry["source"],bytes=entry["bytes"],SHA256=entry["SHA256"],
                                   differing_fields=[key for key in STAT_FIELDS if seen[key]!=old[leaf][key]]))
        else:
            rows[leaf] = dict(source=entry["source"],bytes=entry["bytes"],SHA256=entry["SHA256"],stamp=seen)
    if mismatched:
        return dict(state="OLD361_IDENTITY_CHANGED_NEEDS_TARGETED_REHASH",request=REQUEST,prepare_allowed=False,
                    old361_original_step="1087656.361",matched_old14=len(rows),confirmed_public_shards=len(sealed),
                    required_targeted_rehash=mismatched,untouched_body_bytes=0)
    require(set(rows).isdisjoint(sealed),"Duplicate model member")
    for leaf,row in sealed.items():
        rows[leaf]=row
    require(len(rows)==17 and all(TOKEN.fullmatch(name) for name in rows)
            and sum(row["bytes"] for row in rows.values())==17072712250,"17 original member byte denominator differs")
    index = rows["model.safetensors.index.json"]
    manifest, current = document(index["source"],index["SHA256"],131072,extent=index["bytes"],owner=owner)
    require(current==index["stamp"] and set(manifest["weight_map"].values())==
            {f"model-0000{i}-of-00004.safetensors" for i in range(1,5)},
            "Original HF index does not select the four exact shard filenames")
    return dict(state="SOURCE_IDENTITIES_BOUND",request=REQUEST,label=LABEL,prepare_allowed=True,
                members=rows,modelroot=str(original/"modelroot"),container_root=str(MOUNT_ROOT),
                old361_original_step="1087656.361",logical_index_metadata_bytes=index["bytes"],
                source_weight_body_bytes=0,new_weight_body_bytes=0,hardlinks=0,symlinks=0,
                actual_GPU_steps=0,training_READY=False)


def write_once(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def verify_originals(plan, *, owner=OWNER):
    for row in plan["members"].values():
        observed,_ = fd_info(row["source"],owner=owner)
        require(observed==row["stamp"],"Model source changed after binding")
        if "checkpoint" in row:
            observed,_ = fd_info(row["checkpoint"],owner=owner)
            require(observed==row["checkpoint_stamp"],"Checkpoint source changed after binding")


def prepare(plan, output):
    require(plan["prepare_allowed"] is True and plan["request"] == REQUEST
            and len(plan["members"])==17
            and output == Path(plan["modelroot"]).parent and not output.exists() and not output.is_symlink(),
            "Expected an unattempted, unique v3 output namespace")
    output.mkdir(mode=0o700)
    target = output/"modelroot"
    target.mkdir(mode=0o700)
    for leaf in sorted(plan["members"]):
        write_once(target/leaf,b"")
    binds = [f"{target}:{MOUNT_ROOT}:ro"]
    binds.extend(f"{row['source']}:{MOUNT_ROOT/leaf}:ro" for leaf,row in sorted(plan["members"].items()))
    verify_originals(plan)
    prepared = dict(plan,binds=binds,state="HOST_PLACEHOLDERS_PREPARED_CONTAINER_UNVERIFIED")
    raw = serialized(prepared)
    write_once(output/"MOUNT_PLAN.json",raw)
    return dict(request=REQUEST,state=prepared["state"],plan=str(output/"MOUNT_PLAN.json"),
                plan_SHA256=digest(raw),binds=binds,model_mount=str(MOUNT_ROOT),training_READY=False)


def readonly_mounts(contents):
    mounts = {}
    for line in contents.splitlines():
        left,separator,_ = line.partition(" - ")
        if separator:
            columns = left.split()
            if len(columns)>5:
                path = columns[4]
                mounts[path] = path not in mounts and "ro" in columns[5].split(",")
    return {path for path,readonly in mounts.items() if readonly}


def verify_mounted(plan, mounts, modelroot=MOUNT_ROOT, *, owner=OWNER):
    modelroot = Path(modelroot)
    require(plan["state"]=="HOST_PLACEHOLDERS_PREPARED_CONTAINER_UNVERIFIED"
            and plan["request"]==REQUEST and len(plan["members"])==17
            and set(plan["members"])=={p.name for p in modelroot.iterdir()},
            "Unexpected container model root or member set")
    for leaf,row in plan["members"].items():
        path = modelroot/leaf
        require(str(path) in mounts,"Exact member is not a readonly per-file mount")
        observed,_ = fd_info(path,owner=owner)
        require(observed==row["stamp"],"Mounted inode does not match verified HOST source")
    return dict(request=REQUEST,state="VERIFIED_ALL17_READONLY_REGULAR_MOUNTS",members=17,
                logical_model_bytes=17072712250,model_mount=str(modelroot),source_weight_body_bytes=0,
                training_READY=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("inspect","prepare","verify-mounted","verify-original"))
    p.add_argument("--authority",type=Path)
    p.add_argument("--authority-sha256")
    p.add_argument("--plan",type=Path)
    p.add_argument("--plan-sha256")
    args = p.parse_args()
    require(os.name=="posix" and os.getuid()==OWNER,"Only the project HOST/container owner may apply the plan")
    output = ROOT/LABEL
    if args.mode in ("inspect","prepare"):
        require(args.authority is not None and args.authority_sha256 is not None,
                "Provide the adopted exact digest of the three independent terminal seal bindings")
        assets,_ = document(HERE/"ASSETS.json",ASSETS_SHA,65536)
        authority,_ = document(args.authority,args.authority_sha256,65536)
        outcome,_ = document(ORIGINAL_361,OUTCOME_SHA,OUTCOME_BYTES,extent=OUTCOME_BYTES)
        result = build_plan(assets,authority,output,outcome)
        if args.mode=="prepare":
            claim = json.loads((HERE/"CLAIM.json").read_bytes())
            require(claim["request"]==REQUEST and claim["CPU_GO"] is True and claim["GPU_GO"] is False
                    and args.authority_sha256==claim["adopted_seals_SHA256"]
                    and claim["adopted_public_shard02_checkpoint_SHA256"]==authority["seals"]["2"]["checkpoint_SHA256"],
                    "Adopted three-seal digest required for exclusive namespace preparation")
            result=prepare(result,output)
    else:
        require(args.plan is not None and args.plan_sha256 is not None,"Supply the same immutable original mount plan")
        plan,_ = document(args.plan,args.plan_sha256,65536)
        if args.mode=="verify-mounted":
            result=verify_mounted(plan,readonly_mounts(Path("/proc/self/mountinfo").read_text()))
        else:
            verify_originals(plan)
            result=dict(request=REQUEST,state="ORIGINAL_17_AND_SEAL_METADATA_STAMPS_UNCHANGED",members=17)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
