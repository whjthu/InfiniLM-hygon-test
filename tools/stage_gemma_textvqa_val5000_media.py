"""Stage bounded, answerless batches from the sealed local TextVQA media manifest."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4
import zipfile

from stage_sdxl_official_text_encoder import document, save, send, transport


MANIFEST_SHA = "388a384f576c2e7d599f5475a7ff44fbdc7b6eebfec9a4581168184b7e099f3e"
ASSET_BYTES = 916425313
BATCH_BYTES = 12 << 20
PLAN_SHA = "7ca159f8d2268b836b5d641c190bea7fd01fd5c32dfeff1fe70f47019e9469e9"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def load_manifest(path):
    with path.open("rb") as stream:
        raw = stream.read((2 << 20) + 1)
    need(0 < len(raw) <= 2 << 20 and digest(raw) == MANIFEST_SHA, "Exact canonical media manifest required")
    rows = json.loads(raw)
    ids = [row["image_id"] for row in rows]
    need(len(ids) == len(set(ids)) == len(set(row["SHA256"] for row in rows)) == 3166 and sorted(ids) == ids
         and sum(row["bytes"] for row in rows) == ASSET_BYTES, "Canonical media denominator differs")
    return rows


def batches(rows):
    result, current, extent = [], [], 0
    for row in rows:
        size = row["bytes"]
        need(type(size) is int and 0 < size <= 6 << 20, "Source media extent differs")
        if extent + size > BATCH_BYTES:
            result.append(current)
            current, extent = [], 0
        current.append(row)
        extent += size
    result.append(current)
    return result


def public_row(row):
    return dict(name=row["image_id"] + ".jpg", bytes=row["bytes"], SHA256=row["SHA256"])


def packet(index, rows, root):
    roster = [public_row(row) for row in rows]
    metadata = document(dict(manifest_SHA256=MANIFEST_SHA, index=index, rows=roster))
    memory = BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("BATCH.json", metadata)
        for original, binding in zip(rows, roster, strict=True):
            path = root / original["relative_local_path"]
            need(path.resolve(strict=True) == path and path.is_relative_to(root) and path.name == binding["name"],
                 "Canonical local media path differs")
            with path.open("rb") as stream:
                raw = stream.read(binding["bytes"] + 1)
            need(len(raw) == binding["bytes"] and digest(raw) == binding["SHA256"], "Actual local media bytes differ")
            archive.writestr(binding["name"], raw)
    return memory.getvalue(), digest(metadata)


REMOTE = r"""
import hashlib,io,json,os,re,sys,zipfile
from pathlib import Path

def need(ok, why):
    if not ok: raise ValueError(why)

raw=sys.stdin.buffer.read((13<<20)+1)
need(0<len(raw)<=13<<20,"Exact bounded archive length")
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming')/('gemma-textvqa-val5000-media-'+MANIFEST[:16])
root.mkdir(mode=0o700,exist_ok=True)
need(root.resolve(strict=True)==root and root.stat().st_uid==os.getuid(),"Exact media owner directory")
with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
    members=bundle.namelist()
    need(len(members)==len(set(members)) and 'BATCH.json' in members,"Exact unique ZIP members")
    info=bundle.getinfo('BATCH.json')
    need(0<info.file_size<=64<<10 and info.compress_type==zipfile.ZIP_STORED,"Exact public roster member")
    content=bundle.read(info)
    need(hashlib.sha256(content).hexdigest()==ROSTER,"Exact public roster binding")
    roster=json.loads(content)
    rows=roster['rows']
    pins={v['name']:v for v in rows}
    need(roster['manifest_SHA256']==MANIFEST and roster['index']==INDEX and 0<len(rows)==len(pins)
         and set(members)=={'BATCH.json',*pins} and all(re.fullmatch(r'[a-f0-9]{16}\.jpg',name) for name in pins)
         and sum(p['bytes'] for p in rows)<=12<<20,"Exact answerless media closure")
    target=root/f'batch-{INDEX:03d}'
    pending=root/f'batch-{INDEX:03d}.incomplete'
    need(not target.exists() and not pending.exists(),"Original batch is complete or uncertain")
    pending.mkdir(mode=0o700)
    for name in sorted(pins):
        p=pins[name]
        item=bundle.getinfo(name)
        need(0<p['bytes']==item.file_size<=6<<20 and item.compress_type==zipfile.ZIP_STORED,"Exact media extent")
        image=bundle.read(item)
        need(hashlib.sha256(image).hexdigest()==p['SHA256'],"Actual original image differs")
        path=pending/name
        with path.open('xb') as stream:
            stream.write(image);stream.flush();os.fsync(stream.fileno())
        need(hashlib.sha256(path.read_bytes()).hexdigest()==p['SHA256'],"Actual target readback differs")
    result={'rc':0,'request':REQUEST,'index':INDEX,'manifest_SHA256':MANIFEST,'roster_SHA256':ROSTER,
            'verified_target_images':len(rows),'verified_target_bytes':sum(p['bytes'] for p in rows),
            'answer_rows_transferred':0}
    for name,value in {'BATCH.json':content,'RECEIPT.json':(json.dumps(result,sort_keys=True)+'\n').encode()}.items():
        with (pending/name).open('xb') as stream:
            stream.write(value);stream.flush();os.fsync(stream.fileno())
    os.rename(pending,target)
    parent=os.open(root,os.O_DIRECTORY)
    try: os.fsync(parent)
    finally: os.close(parent)
print(json.dumps(result,sort_keys=True))
"""


STATUS = r"""
import hashlib,json,sys
from pathlib import Path

expected=json.load(sys.stdin)
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming')/('gemma-textvqa-val5000-media-'+MANIFEST[:16])
complete=[]
incomplete=[]
absent=[]
bad=[]
images=size=0
for row in expected:
    index=row['index']
    final=root/f'batch-{index:03d}'
    pending=final.with_suffix('.incomplete')
    if pending.is_dir():
        incomplete.append(index)
    elif not final.is_dir():
        absent.append(index)
    else:
        try:
            roster_file=final/'BATCH.json'
            receipt_file=final/'RECEIPT.json'
            with roster_file.open('rb') as stream: raw=stream.read((64<<10)+1)
            with receipt_file.open('rb') as stream: receipt=json.loads(stream.read(8193))
            files=json.loads(raw)['rows']
            if (not 0<len(raw)<=64<<10 or hashlib.sha256(raw).hexdigest()!=row['roster_SHA256']
                or receipt['manifest_SHA256']!=MANIFEST or receipt['index']!=index or receipt['rc']!=0
                or receipt['roster_SHA256']!=row['roster_SHA256'] or receipt['verified_target_images']!=row['images']
                or receipt['verified_target_bytes']!=row['media_bytes'] or receipt['answer_rows_transferred']!=0
                or {p.name for p in final.iterdir()}!={'BATCH.json','RECEIPT.json',*(f['name'] for f in files)}
                or any((final/f['name']).stat().st_size!=f['bytes'] for f in files)):
                raise ValueError('Published batch evidence changed')
        except (OSError,TypeError,ValueError,KeyError):
            bad.append(index)
        else:
            complete.append(index)
            images+=row['images']
            size+=row['media_bytes']
print(json.dumps({'rc':0,'request':REQUEST,'manifest_SHA256':MANIFEST,'completed':complete,'incomplete':incomplete,
                  'absent':absent,'failed_verification':bad,'verified_target_images':images,
                  'verified_target_bytes':size},sort_keys=True))
"""


def frozen_plan(directory):
    raw = (directory / "PLAN.json").read_bytes()
    need(digest(raw) == PLAN_SHA, "Frozen transfer plan differs")
    return raw


def target_status(directory, prefix, request):
    outcome = send(transport(), request=request,
                   program=f"MANIFEST={MANIFEST_SHA!r}\nREQUEST={request!r}\n" + STATUS,
                   data=frozen_plan(directory), send_token=prefix.with_suffix(".send.json"), outcome_seconds=75)
    save(prefix.with_suffix(".transport.json"), document(outcome))
    return outcome


def unsent_original(directory, index, roster_sha):
    original = directory / f"batch-{index:03d}"
    intent = json.loads(original.with_suffix(".intent.json").read_bytes())
    raw = original.with_suffix(".transport.json").read_bytes()
    prior = json.loads(raw)
    need(intent["index"] == index and intent["roster_SHA256"] == roster_sha and intent["request"] == prior["request"]
         and prior["remote_outcome"] == "PAYLOAD_NOT_SENT" and prior["authority_verified"] is False
         and prior["payload_send_attempted"] is False and prior["payload_flushed"] is False
         and prior["response"] is None, "Original batch transport did not prove zero payload attempts")
    return dict(original_request=intent["request"], original_transport_SHA256=digest(raw))


def check_absent(directory, prefix, index):
    outcome = target_status(directory, prefix, uuid4().hex)
    observed = outcome["response"]
    need(outcome["remote_outcome"] == "RESPONSE_OBSERVED" and observed["rc"] == 0
         and observed["request"] == outcome["request"] and observed["manifest_SHA256"] == MANIFEST_SHA
         and index in observed["absent"] and all(index not in observed[key]
             for key in ("completed", "incomplete", "failed_verification")),
         "Fresh target check did not confirm the original batch is absent")
    return dict(target_check_request=outcome["request"], target_check_SHA256=digest(document(outcome)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "transfer", "status", "retry-unsent"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    planned = batches(load_manifest(args.manifest))
    if args.mode == "plan":
        args.out.mkdir()
        plan = [dict(index=i, images=len(rows), media_bytes=sum(r["bytes"] for r in rows),
                     roster_SHA256=digest(document(dict(manifest_SHA256=MANIFEST_SHA, index=i,
                                                        rows=list(map(public_row, rows))))))
                for i, rows in enumerate(planned)]
        save(args.out / "PLAN.json", document(plan))
        print(json.dumps(dict(batches=len(planned), images=3166, bytes=ASSET_BYTES,
                              largest_batch=max(row["media_bytes"] for row in plan),
                              plan_SHA256=digest(document(plan))), sort_keys=True))
        return 0
    if args.mode == "status":
        request = uuid4().hex
        outcome = target_status(args.out, args.out / f"status-{request}", request)
        print(json.dumps({key: outcome[key] for key in ("request", "remote_outcome", "response")}, sort_keys=True))
        return 0 if outcome["remote_outcome"] == "RESPONSE_OBSERVED" else 2
    index = args.batch
    need(type(index) is int and 0 <= index < len(planned), "Select an exact frozen batch")
    request = uuid4().hex
    body, roster_sha = packet(index, planned[index], args.local_root.resolve(strict=True))
    frozen = json.loads(frozen_plan(args.out))[index]
    need(frozen["index"] == index and frozen["roster_SHA256"] == roster_sha, "Frozen batch plan differs")
    prefix = args.out / f"batch-{index:03d}"
    intent = dict(request=request, index=index, roster_SHA256=roster_sha)
    if args.mode == "retry-unsent":
        prefix = args.out / f"batch-{index:03d}-retry1"
        need(not prefix.with_suffix(".intent.json").exists(), "One recovery attempt already exists")
        intent.update(unsent_original(args.out, index, roster_sha))
        intent.update(check_absent(args.out, args.out / f"batch-{index:03d}-retry-check-{request}", index))
    save(prefix.with_suffix(".intent.json"), document(intent))
    program = f"MANIFEST={MANIFEST_SHA!r}\nROSTER={roster_sha!r}\nREQUEST={request!r}\nINDEX={index}\n" + REMOTE
    outcome = send(transport(), request=request, program=program, data=body,
                   send_token=prefix.with_suffix(".send.json"), packet_cap=28 << 20, write_seconds=110,
                   authority_seconds=22, outcome_seconds=90)
    save(prefix.with_suffix(".transport.json"), document(outcome))
    print(json.dumps({key: outcome[key] for key in ("request", "remote_outcome", "response")}, sort_keys=True))
    return 0 if outcome["remote_outcome"] == "RESPONSE_OBSERVED" and outcome["response"]["rc"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
