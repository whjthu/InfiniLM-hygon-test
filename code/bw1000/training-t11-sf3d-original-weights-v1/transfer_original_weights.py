"""Copy the two pinned original SF3D files through finite existing SSH sessions."""

import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

PINS = {
    "config.yaml": (2405, "5e880cfec3dec28d4e20afe07bfa1799c8a561f56f0532ebf1b4022ae4ddc8e4"),
    "model.safetensors": (4024289892, "a3416e1cf654e7d4f5e75f116cec2c3f0a14501a77d30c2f6068bbda178de388"),
}

SENDER = r'''
import hashlib,json,signal,sys
from pathlib import Path

signal.alarm(900)
root=Path("/mnt/data/nvidia-baseline-20261010/inputs/models/stable-fast-3d")
pins=PINS
stream=sys.stdout.buffer
for name,(size,digest) in pins.items():
    path=root/name
    assert path.is_file() and path.stat().st_size==size
    stream.write((json.dumps({"name":name,"bytes":size,"SHA256":digest})+"\n").encode())
    stream.flush()
    hashed=hashlib.sha256()
    with path.open("rb") as source:
        remaining=size
        while remaining:
            data=source.read(min(1024*1024,remaining))
            assert data
            stream.write(data)
            hashed.update(data)
            remaining-=len(data)
        assert source.read(1)==b""
    stream.flush()
    assert hashed.hexdigest()==digest
'''

RECEIVER = r'''
import datetime,hashlib,json,os,shutil,signal,sys,time
from pathlib import Path

signal.alarm(900)
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-t11-sf3d-original-weights-v1")
assert not root.exists()
assert shutil.disk_usage(root.parent).free>4300000000
root.mkdir(mode=0o700)
started=time.monotonic()
intent={"claim_commit":CLAIM,"state":"COPYING","expected":PINS,
        "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"GPU_work":False}
with (root/"intent.json").open("x") as f:
    json.dump(intent,f)
rows={}
try:
    for name,(size,digest) in PINS.items():
        header=sys.stdin.buffer.readline(512)
        assert header.endswith(b"\n") and json.loads(header)=={"name":name,"bytes":size,"SHA256":digest}
        temp=root/(name+".partial")
        hashed=hashlib.sha256()
        with temp.open("xb") as dest:
            remaining=size
            while remaining:
                data=sys.stdin.buffer.read(min(1024*1024,remaining))
                assert data
                dest.write(data)
                hashed.update(data)
                remaining-=len(data)
            dest.flush()
            os.fsync(dest.fileno())
        assert temp.stat().st_size==size and hashed.hexdigest()==digest
        temp.rename(root/name)
        rows[name]={"bytes":size,"SHA256":digest}
    assert sys.stdin.buffer.read(1)==b""
    result={"claim_commit":CLAIM,"state":"VERIFIED_ORIGINAL_WEIGHTS",
            "files":rows,"wall_seconds":time.monotonic()-started,"GPU_work":False}
except BaseException as error:
    result={"claim_commit":CLAIM,"state":"COPY_FAILED","completed_files":rows,
            "error_type":type(error).__name__,"GPU_work":False}
    with (root/"failure.json").open("x") as f:
        json.dump(result,f)
    raise
with (root/"receipt.json").open("x") as f:
    json.dump(result,f,indent=2)
print(json.dumps(result))
'''


def main():
    base=Path(__file__).resolve().parent
    claim=sys.argv[1]
    assert len(claim)==40 and all(c in "0123456789abcdef" for c in claim)
    prelude=(base.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf-8")
    namespace={"__file__":str(Path(__file__).resolve())}
    exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),namespace)
    cluster=list(namespace["ssh"])
    assert cluster[-1]=="/usr/local/python3.12/bin/python3 -B -u -"
    receiver=f"PINS={PINS!r}\nCLAIM={claim!r}\n"+RECEIVER
    cluster[-1]="/usr/local/python3.12/bin/python3 -B -u -c "+shlex.quote(receiver)
    config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
    sender=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),
            "a100-26","python3 -B -u -c "+shlex.quote(f"PINS={PINS!r}\n"+SENDER)]
    intent={"claim_commit":claim,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope":"finite A100-to-cluster original SF3D asset transfer","expected":PINS}
    with (base/"original-weight-transfer-intent.json").open("x",encoding="utf-8") as f:
        json.dump(intent,f,indent=2)
    started=time.monotonic()
    with (base/"original-weight-sender.stderr").open("xb") as source_error, \
         (base/"original-weight-receiver.stderr").open("xb") as dest_error, \
         (base/"original-weight-receiver.stdout").open("xb") as dest_output:
        destination=subprocess.Popen(cluster,stdin=subprocess.PIPE,stdout=dest_output,stderr=dest_error)
        source=subprocess.Popen(sender,stdout=subprocess.PIPE,stderr=source_error)
        copied=0
        try:
            while True:
                chunk=source.stdout.read(1024*1024)
                if not chunk:
                    break
                destination.stdin.write(chunk)
                copied+=len(chunk)
            destination.stdin.close()
            source_rc=source.wait(timeout=30)
            dest_rc=destination.wait(timeout=30)
        except BaseException:
            source.terminate()
            destination.terminate()
            source.wait(timeout=10)
            destination.wait(timeout=10)
            raise
    raw=(base/"original-weight-receiver.stdout").read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.startswith(b"{")]
    result={"claim_commit":claim,"source_rc":source_rc,"destination_rc":dest_rc,
            "stream_bytes":copied,"wall_seconds":time.monotonic()-started,
            "structured_rows":len(rows),"receipt":rows[0] if len(rows)==1 else None}
    with (base/"original-weight-transfer-outcome.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))
    assert source_rc==dest_rc==0 and len(rows)==1
    assert rows[0]["state"]=="VERIFIED_ORIGINAL_WEIGHTS"


if __name__=="__main__":
    main()
