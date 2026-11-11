"""Export only the exact original COCO64 and two source plugins from A100."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
plan=json.load(sys.stdin)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode="w") as archive:
    for item in plan["files"]:
        raw=Path(item["source"]).read_bytes()
        assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["SHA256"], item["name"]
        info=tarfile.TarInfo(item["name"]);info.size=len(raw);info.mode=0o600
        archive.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(buf.getvalue())
