"""Export exact non-weight processor files and private reference identities."""
import gzip,hashlib,io,json,sys,tarfile
from pathlib import Path
plan=json.load(sys.stdin)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode="w:gz") as archive:
    for item in plan["files"]:
        assert not item["name"].endswith((".safetensors",".pt"))
        raw=Path(item["source"]).read_bytes()
        assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["SHA256"],item["name"]
        info=tarfile.TarInfo(item["name"]);info.size=len(raw);info.mode=0o600
        archive.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(buf.getvalue())
