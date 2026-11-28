"""Stage one official SDXL text encoder in independently acknowledged parts."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from run_hygon_bound_payload import digest, document, send


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
ASSETS = LOCAL / "sdxl-official-text-encoders-4621659-20261128"
ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming"
REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
BLOCK = 4 << 20
TRANSPORT = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
ASSET_RECEIPT = "9d8e503524c493760405df62dd7e7f5be43b0aa47312bfec131eb1b5814234c3"
PINNED = {
    "text_encoder": (492_265_168, "5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc"),
    "text_encoder_2": (2_778_702_264, "3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def save(path, raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def prepare(tower):
    receipt = (ASSETS / "receipt.json").read_bytes()
    require(digest(receipt) == ASSET_RECEIPT, "Official download receipt differs")
    rows = json.loads(receipt)["files"]
    config = (ASSETS / tower / "config.json").read_bytes()
    cfg = rows[f"{tower}/config.json"]
    require(len(config) == cfg["bytes"] and digest(config) == cfg["SHA256"], "Official config differs")
    path = ASSETS / tower / "model.safetensors"
    size, expected = PINNED[tower]
    whole = sha256()
    parts, pos = [], 0
    with path.open("rb") as source:
        for index, data in enumerate(iter(lambda: source.read(BLOCK), b"")):
            whole.update(data)
            parts.append(dict(index=index, offset=pos, bytes=len(data), sha256=digest(data)))
            pos += len(data)
    require(pos == size == path.stat().st_size and whole.hexdigest() == expected, "Official weight bytes differ")
    key = f"sdxl-{tower.replace('_', '-')}-official-4621659-v1"
    plan = document(dict(namespace=key, revision=REVISION, tower=tower, bytes=size,
                         sha256=expected, config_sha256=digest(config), chunks=parts))
    return path, config, plan


BASE = r"""
from hashlib import sha256
import json,os,stat,sys
from pathlib import Path
dest=Path(ROOT)/NAMESPACE
def pin(data):
 return sha256(data).hexdigest()
def save(path,data):
 with path.open('xb') as f:
  f.write(data);f.flush();os.fsync(f.fileno())
def doc(value):
 return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def reply(**kwargs):
 print(json.dumps(dict(rc=0,request=REQUEST,plan_sha256=PLAN_PIN,**kwargs),sort_keys=True),flush=True)
"""

INIT = r"""
from base64 import b64decode
packet=json.load(sys.stdin)
plan=b64decode(packet['plan'],validate=True)
config=b64decode(packet['config'],validate=True)
assert 0<len(plan)<=262144 and pin(plan)==PLAN_PIN
expected=json.loads(plan)
assert expected['namespace']==NAMESPACE and pin(config)==expected['config_sha256']
os.umask(0o077)
dest.mkdir(mode=0o700)
save(dest/'config.json',config)
save(dest/'plan.json',plan)
reply(state='INIT',parts=len(expected['chunks']))
"""

PART = r"""
plan=(dest/'plan.json').read_bytes()
assert 0<len(plan)<=262144 and pin(plan)==PLAN_PIN
row=json.loads(plan)['chunks'][INDEX]
assert row['index']==INDEX and 0<row['bytes']<=4<<20
raw=sys.stdin.buffer.read((4<<20)+1)
assert len(raw)==row['bytes'] and pin(raw)==row['sha256']
part=dest/f'part-{INDEX:03d}.bin'
receipt=dest/f'part-{INDEX:03d}.json'
partial=dest/f'part-{INDEX:03d}.incomplete'
assert not any(os.path.lexists(p) for p in (part,receipt,partial))
os.umask(0o077)
save(partial,raw)
assert pin(partial.read_bytes())==row['sha256'] and not os.path.lexists(part)
os.rename(partial,part)
values=dict(state='PART',index=INDEX,bytes=len(raw),sha256=row['sha256'])
save(receipt,doc(dict(request=REQUEST,plan_sha256=PLAN_PIN,**values)))
reply(**values)
"""


ASSEMBLE = r"""
plan_bytes=(dest/'plan.json').read_bytes()
assert 0<len(plan_bytes)<=262144 and pin(plan_bytes)==PLAN_PIN
plan=json.loads(plan_bytes)
expected=json.load(sys.stdin)
assert plan['namespace']==NAMESPACE and len(plan['chunks'])==expected['parts']
assert (plan['tower'],plan['revision'],plan['bytes'],plan['sha256'])==(expected['tower'],expected['revision'],expected['bytes'],expected['sha256'])
config=(dest/'config.json').read_bytes()
assert pin(config)==plan['config_sha256']
published=dest/'model.safetensors'
partial=dest/'model.safetensors.assembly-incomplete'
intent=dest/'assembly-intent.json'
seal=dest/'assembly-receipt.json'
assert not any(os.path.lexists(p) for p in (published,partial,intent,seal))
parts=[]
position=0
for index,row in enumerate(plan['chunks']):
 assert row['index']==index and row['offset']==position and 0<row['bytes']<=4<<20
 part=dest/f'part-{index:03d}.bin'
 receipt=dest/f'part-{index:03d}.json'
 assert not os.path.lexists(dest/f'part-{index:03d}.incomplete')
 assert part.resolve(strict=True)==part and stat.S_ISREG(part.stat().st_mode) and part.stat().st_size==row['bytes']
 assert receipt.resolve(strict=True)==receipt
 ack=json.loads(receipt.read_bytes())
 assert (ack['plan_sha256'],ack['state'],ack['index'],ack['bytes'],ack['sha256'])==(PLAN_PIN,'PART',index,row['bytes'],row['sha256'])
 parts.append((part,row))
 position+=row['bytes']
assert position==plan['bytes']
os.umask(0o077)
save(intent,doc(dict(request=REQUEST,plan_sha256=PLAN_PIN,parts=len(parts),bytes=position)))
whole=sha256()
with partial.open('xb') as output:
 for part,row in parts:
  with part.open('rb') as source:
   raw=source.read(row['bytes']+1)
  assert len(raw)==row['bytes'] and pin(raw)==row['sha256']
  output.write(raw)
  whole.update(raw)
 output.flush()
 os.fsync(output.fileno())
assert partial.stat().st_size==plan['bytes'] and whole.hexdigest()==plan['sha256']
reread=sha256()
with partial.open('rb') as source:
 for raw in iter(lambda:source.read(4<<20),b''):
  reread.update(raw)
assert reread.hexdigest()==whole.hexdigest()
os.link(partial,published)
partial.unlink()
values=dict(state='PUBLISHED',parts=len(parts),bytes=position,sha256=whole.hexdigest(),config_sha256=pin(config),whole_readback=True)
save(seal,doc(dict(request=REQUEST,plan_sha256=PLAN_PIN,**values)))
reply(**values)
"""


def transport():
    path = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = path.read_bytes()
    require(digest(raw) == TRANSPORT, "Existing transport helper differs")
    client = ModuleType("original_transport")
    client.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), vars(client))
    return client.ssh(REPO.parent)


def transmit(ssh, root, namespace, plan_sha, suffix, data, *, label, index=None, outcome_seconds=40):
    for _ in range(2):
        request = uuid4().hex
        bindings = dict(ROOT=ROOT, NAMESPACE=namespace, REQUEST=request, PLAN_PIN=plan_sha, INDEX=index)
        source = "\n".join(f"{key}={value!r}" for key, value in bindings.items()) + "\n" + BASE + suffix
        result = send(ssh, request=request, program=source, data=data, packet_cap=10 << 20,
                      send_token=root / f"{label}-{request}-send.json", authority_seconds=22,
                      write_seconds=100, outcome_seconds=outcome_seconds)
        save(root / f"{label}-{request}.json", document(result))
        print(json.dumps(dict(label=label, request=request, outcome=result["remote_outcome"],
                              response=result["response"]), sort_keys=True), flush=True)
        if result["remote_outcome"] == "RESPONSE_OBSERVED":
            response = result["response"]
            require(response["rc"] == 0 and response["plan_sha256"] == plan_sha, "Original receiver result differs")
            return response
        if result["payload_send_attempted"] is not False:
            raise RuntimeError(f"Original payload requires reconciliation for {label}")
    raise RuntimeError(f"No original payload was sent for {label}")


def assemble(tower):
    namespace = f"sdxl-{tower.replace('_', '-')}-official-4621659-v1"
    root = ASSETS / namespace
    plan = (root / "plan.json").read_bytes()
    actual = json.loads(plan)
    size, expected = PINNED[tower]
    require((actual["namespace"], actual["tower"], actual["revision"], actual["bytes"], actual["sha256"])
            == (namespace, tower, REVISION, size, expected), "Assembly requires the original official plan")
    pin = digest(plan)
    parts = len(actual["chunks"])
    require(json.loads((root / "all-parts-acknowledged.json").read_bytes()) == dict(plan_sha256=pin, parts=parts),
            "All original upload acknowledgements are required before assembly")
    response = transmit(transport(), root, namespace, pin, ASSEMBLE,
                        document(dict(tower=tower, revision=REVISION, bytes=size, sha256=expected, parts=parts)),
                        label="assembly", outcome_seconds=300)
    require((response["state"], response["parts"], response["bytes"], response["sha256"], response["whole_readback"])
            == ("PUBLISHED", parts, size, expected, True), "Target assembly receipt mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tower", choices=PINNED)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true")
    mode.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        assemble(args.tower)
        return
    path, cfg, plan = prepare(args.tower)
    actual = json.loads(plan)
    pin = digest(plan)
    root = ASSETS / actual["namespace"]
    print(json.dumps(dict(plan_sha256=pin, bytes=actual["bytes"], parts=len(actual["chunks"]),
                          source_sha256=actual["sha256"], network=args.send), sort_keys=True), flush=True)
    if not args.send:
        return
    root.mkdir(mode=0o700)
    save(root / "plan.json", plan)
    ssh = transport()
    init = transmit(ssh, root, actual["namespace"], pin, INIT,
                    document(dict(plan=b64encode(plan).decode(), config=b64encode(cfg).decode())), label="init")
    require((init["state"], init["parts"]) == ("INIT", len(actual["chunks"])), "Original receiver init mismatch")
    with path.open("rb") as file:
        for row in actual["chunks"]:
            raw = file.read(row["bytes"])
            require(file.tell() == row["offset"] + row["bytes"] and digest(raw) == row["sha256"],
                    "Frozen source part differs")
            index = row["index"]
            response = transmit(ssh, root, actual["namespace"], pin, PART, raw,
                                label=f"part-{index:03d}", index=index)
            require((response["state"], response["index"], response["bytes"], response["sha256"]) ==
                    ("PART", index, len(raw), row["sha256"]), "Original part receipt mismatch")
    save(root / "all-parts-acknowledged.json", document(dict(plan_sha256=pin, parts=len(actual["chunks"]))))


if __name__ == "__main__":
    main()
