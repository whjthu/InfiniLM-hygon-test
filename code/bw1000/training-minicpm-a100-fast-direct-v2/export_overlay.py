"""Export only SHA-pinned original fast-runtime software files."""
import hashlib,io,json,signal,sys,tarfile,time
from pathlib import Path
remaining=DEADLINE-time.time()
assert 0<remaining<=600
signal.alarm(max(1,int(remaining)))
raw=sys.stdin.buffer.read(2000000);assert len(raw)<2000000
assert hashlib.sha256(raw).hexdigest()==PLAN_SHA
plan=json.loads(raw);buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode="w:gz") as archive:
    info=tarfile.TarInfo("PLAN.json");info.size=len(raw);info.mode=0o600
    archive.addfile(info,io.BytesIO(raw))
    for name,item in plan["files"].items():
        assert name.startswith(("site/","bin/")) and not name.endswith((".pt",".safetensors",".txt.source-sha256"))
        p=Path(item["source"]);data=p.read_bytes()
        assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["SHA256"],name
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=item["mode"]&0o777
        archive.addfile(info,io.BytesIO(data))
assert buffer.tell()<256*1024**2
sys.stdout.buffer.write(buffer.getvalue())
