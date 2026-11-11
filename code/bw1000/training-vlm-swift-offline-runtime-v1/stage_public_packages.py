"""Stage exact public PyPI package files through the existing cluster SSH connection."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import urllib.parse
import urllib.request

RECEIVER = r'''
import hashlib,json,signal,sys,tarfile,time
from pathlib import Path
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vlm-swift-public-artifacts-v1")
signal.alarm(600)
assert not root.exists()
root.mkdir(mode=0o700)
(root/"packages").mkdir(mode=0o700)
with (root/"intent.json").open("x") as f:json.dump({"claim_commit":CLAIM,"expected":PINS,"public_python_packages":True},f)
rows={}
started=time.monotonic()
with tarfile.open(fileobj=sys.stdin.buffer,mode="r|") as archive:
    for member in archive:
        assert member.isfile() and member.name in PINS and Path(member.name).name==member.name and member.name not in rows
        expected=PINS[member.name]
        assert member.size==expected["bytes"] and 0<member.size<100*1024**2
        source=archive.extractfile(member)
        hashed=hashlib.sha256()
        temporary=root/"packages"/(member.name+".partial")
        with temporary.open("xb") as f:
            remaining=member.size
            while remaining:
                data=source.read(min(1024*1024,remaining))
                assert data
                f.write(data)
                hashed.update(data)
                remaining-=len(data)
        assert temporary.stat().st_size==member.size and hashed.hexdigest()==expected["SHA256"]
        temporary.rename(root/"packages"/member.name)
        rows[member.name]=expected
assert set(rows)==set(PINS)
result={"claim_commit":CLAIM,"files":rows,"total_bytes":sum(r["bytes"] for r in rows.values()),
        "wall_seconds":time.monotonic()-started,"state":"VERIFIED_PUBLIC_PACKAGE_FILES"}
with (root/"receipt.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("repo",type=Path)
    parser.add_argument("claim")
    parser.add_argument("artifacts",type=Path)
    options=parser.parse_args()
    base=Path(__file__).resolve().parent
    def blob(name):
        return subprocess.check_output(["git","-c","safe.directory="+options.repo.as_posix(),"-C",str(options.repo),
                                        "show",options.claim+":"+name])
    task=json.loads(blob("coordination/tasks/training-vlm-swift-offline-runtime-v1.json"))
    raw=blob(task["source"]+"/PUBLIC_PACKAGES.json")
    assert hashlib.sha256(raw).hexdigest()==task["source_SHA256"]["PUBLIC_PACKAGES.json"]
    manifest=json.loads(raw)
    pins={}
    for row in manifest["packages"]:
        filename=row["filename"]
        assert Path(filename).name==filename and filename not in pins
        assert urllib.parse.urlsplit(row["url"]).hostname in ("pypi.org","files.pythonhosted.org")
        path=options.artifacts/filename
        if not path.exists():
            options.artifacts.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(row["url"],timeout=30) as response:
                data=response.read(row["bytes"]+1)
            assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["SHA256"]
            with path.open("xb") as stream:stream.write(data)
        hashed=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                hashed.update(chunk)
        assert path.stat().st_size==row["bytes"] and hashed.hexdigest()==row["SHA256"]
        pins[filename]={"bytes":row["bytes"],"SHA256":row["SHA256"]}
    prelude=(base.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf-8")
    namespace={"__file__":str(Path(__file__).resolve())}
    exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),namespace)
    command=list(namespace["ssh"])
    assert command[-1]=="/usr/local/python3.12/bin/python3 -B -u -"
    remote=f"CLAIM={options.claim!r}\nPINS={pins!r}\n"+RECEIVER
    command[-1]="/usr/local/python3.12/bin/python3 -B -u -c "+shlex.quote(remote)
    with (base/"public-stage-intent.json").open("x") as stream:
        json.dump({"claim_commit":options.claim,"total_bytes":sum(r["bytes"] for r in pins.values())},stream)
    with (base/"public-stage.stdout").open("xb") as out, (base/"public-stage.stderr").open("xb") as error:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=error)
        try:
            with tarfile.open(fileobj=child.stdin,mode="w|") as archive:
                for name in pins:
                    archive.add(options.artifacts/name,arcname=name,recursive=False)
            child.stdin.close()
            rc=child.wait(timeout=120)
        except BaseException:
            child.terminate()
            child.wait(timeout=10)
            raise
    raw=(base/"public-stage.stdout").read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.startswith(b"{")]
    result={"transport_rc":rc,"structured_rows":len(rows),"receipt":rows[0] if len(rows)==1 else None}
    with (base/"public-stage-outcome.json").open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({"transport_rc":rc,"structured_rows":len(rows),
                      "file_count":len(rows[0]["files"]) if rows else None,"total_bytes":rows[0]["total_bytes"] if rows else None}))
    assert rc==0 and len(rows)==1 and rows[0]["files"]==pins

if __name__=="__main__":
    main()
